import os
import argparse
import torch
import torch.distributed as dist
from model.VRP_encoder import VRP_encoder
from SAM2pred import SAM_pred
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset


def validate(args, epoch, model, sam_model, dataloader):
    """Validation function directly taken from the original training script."""
    print(f"Starting validation for epoch {epoch}...")
    utils.fix_randseed(args.seed + epoch)
    model.module.eval()
    average_meter = AverageMeter(dataloader.dataset)

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            if idx % 10 == 0:
                print(f"Validating batch {idx + 1}/{len(dataloader)}...")
            batch = utils.to_cuda(batch)
            protos, _ = model(args.condition, batch['query_img'], batch['support_imgs'].squeeze(1),
                              batch['support_masks'].squeeze(1), training=False)

            low_masks, pred_mask = sam_model(batch['query_img'], batch['query_name'], protos)
            logit_mask = low_masks
            pred_mask = torch.sigmoid(logit_mask) > 0.5
            pred_mask = pred_mask.float()

            loss = model.module.compute_objective(logit_mask, batch['query_mask'])

            area_inter, area_union = Evaluator.classify_prediction(pred_mask.squeeze(1), batch)
            average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())

    print("Validation completed for all batches.")
    average_meter.write_result('Validation', epoch)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()

    print(f"Validation Results - Epoch {epoch}: Loss = {avg_loss}, mIoU = {miou}, Fb_IoU = {fb_iou}")
    return avg_loss, miou, fb_iou


def combine_weights_greedy(args, base_model, sam_model, checkpoints, dataloader):
    """Perform greedy model soup, continuing until no improvement is detected."""
    print("Selecting the best initial checkpoint based on stored mIoU values...")
    checkpoint_infos = []

    # Load and store checkpoints with their mIoU values
    for i, ckpt_path in enumerate(checkpoints):
        print(f"Processing checkpoint {i + 1}/{len(checkpoints)}: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        miou = checkpoint.get('best_val_miou', checkpoint.get('val_miou'))
        if miou is None:
            print(f"Skipping checkpoint {ckpt_path}: No 'best_val_miou' or 'val_miou'.")
            continue
        print(f"Checkpoint mIoU: {miou}")
        checkpoint_infos.append({
            'path': ckpt_path,
            'miou': miou,
            'state': checkpoint['model_state_dict']
        })

    if not checkpoint_infos:
        raise RuntimeError("No valid checkpoints found with 'best_val_miou' or 'val_miou'. Please check the input directory.")

    checkpoint_infos.sort(key=lambda x: x['miou'], reverse=True)
    top_checkpoints = checkpoint_infos[:args.top_k]
    print(f"Top {args.top_k} checkpoints selected for combination.")

    best_checkpoint = top_checkpoints[0]
    best_miou = best_checkpoint['miou']
    print(f"Best initial checkpoint: {best_checkpoint['path']} with mIoU: {best_miou}")

    best_model_state = {
        k: v.to(args.local_rank) for k, v in best_checkpoint['state'].items()
    }
    base_model.load_state_dict(best_model_state)

    print("Starting greedy weight combination...")
    improvement = True
    while improvement:
        improvement = False
        for i, checkpoint in enumerate(top_checkpoints):
            if checkpoint['path'] == best_checkpoint['path']:
                print(f"Skipping checkpoint {i + 1}: {checkpoint['path']} (already selected as best).")
                continue

            print(f"Combining weights with checkpoint {i + 1}: {checkpoint['path']}")
            model_state = {
                k: v.to(args.local_rank) for k, v in checkpoint['state'].items()
            }
            combined_state = {
                k: (best_model_state[k] + model_state[k]) / 2
                for k in best_model_state.keys()
            }
            base_model.load_state_dict(combined_state)

            print("Validating combined model...")
            _, miou, _ = validate(args, 0, base_model, sam_model, dataloader)
            print(f"Validation completed. Combined mIoU: {miou}")

            if miou > best_miou:
                print(f"New best model found with mIoU: {miou}")
                best_miou = miou
                best_model_state = combined_state
                best_checkpoint_path = os.path.join(args.target_dir, "best_checkpoint.pth")
                torch.save({
                    'model_state_dict': best_model_state,
                    'best_val_miou': best_miou,
                }, best_checkpoint_path)
                print(f"Best model updated and saved to {best_checkpoint_path}")
                improvement = True  # Mark improvement for another loop

    print("Greedy weight combination completed.")
    base_model.load_state_dict(best_model_state)
    return base_model, best_miou


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Greedy Model Soup')
    parser.add_argument('--source_dir', type=str, required=True, help='Directory with checkpoints')
    parser.add_argument('--target_dir', type=str, required=True, help='Directory to save the model soup')
    parser.add_argument('--datapath', type=str, default='/media/wasfy/KINGSTON/phd/patternRecognition/vrpsam/Datasets_HSN')
    parser.add_argument('--benchmark', type=str, default='coco', choices=['pascal', 'coco', 'fss'])
    parser.add_argument('--logpath', type=str, default='/media/wasfy/KINGSTON/phd/patternRecognition/vrpsam/logs')
    parser.add_argument('--bsz', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--nworker', type=int, default=8)
    parser.add_argument('--seed', type=int, default=321)
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--condition', type=str, default='mask', choices=['point', 'scribble', 'box', 'mask'])
    parser.add_argument('--use_ignore', type=bool, default=True)
    parser.add_argument('--num_query', type=int, default=50)
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['vgg16', 'resnet50', 'resnet101'])
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
    parser.add_argument('--top_k', type=int, default=10, help='Limit processing to top_k checkpoints based on mIoU')
    args = parser.parse_args()

    if args.local_rank == -1:
        args.local_rank = int(os.environ.get('LOCAL_RANK', 0))

    torch.cuda.set_device(args.local_rank)
    device = torch.device(f'cuda:{args.local_rank}')
    dist.init_process_group(backend='nccl', init_method='env://')

    utils.fix_randseed(args.seed)

    FSSDataset.initialize(img_size=512, datapath=args.datapath, use_original_imgsize=False)
    dataloader_val = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'val')

    model = VRP_encoder(args, args.backbone, False).to(device)
    sam_model = SAM_pred().to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    checkpoints = [os.path.join(args.source_dir, f) for f in os.listdir(args.source_dir) if f.endswith('.pth')]

    model, best_miou = combine_weights_greedy(args, model, sam_model, checkpoints, dataloader_val)

    if dist.get_rank() == 0:
        os.makedirs(args.target_dir, exist_ok=True)
        soup_path = os.path.join(args.target_dir, "model_soup.pth")
        torch.save({'model_state_dict': model.state_dict(), 'best_miou': best_miou}, soup_path)
        print(f"Model soup saved to {soup_path}")
