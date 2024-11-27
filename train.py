import os
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from model.VRP_encoder import VRP_encoder
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset
from SAM2pred import SAM_pred


def train(args, epoch, model, sam_model, dataloader, optimizer, scheduler, training):
    """ Train the VRP_encoder model """
    utils.fix_randseed(args.seed + epoch) if training else utils.fix_randseed(args.seed)
    model.module.train_mode() if training else model.module.eval()
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):
        batch = utils.to_cuda(batch)
        protos, _ = model(args.condition, batch['query_img'], batch['support_imgs'].squeeze(1),
                          batch['support_masks'].squeeze(1), training)

        low_masks, pred_mask = sam_model(batch['query_img'], batch['query_name'], protos)
        logit_mask = low_masks
        pred_mask = torch.sigmoid(logit_mask) > 0.5
        pred_mask = pred_mask.float()

        loss = model.module.compute_objective(logit_mask, batch['query_mask'])
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        area_inter, area_union = Evaluator.classify_prediction(pred_mask.squeeze(1), batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())
        average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)

    average_meter.write_result('Training' if training else 'Validation', epoch)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()

    return avg_loss, miou, fb_iou


if __name__ == '__main__':
    # Arguments parsing
    parser = argparse.ArgumentParser(description='Visual Prompt Encoder Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='/root/paddlejob/workspace/env_run/datsets/')
    parser.add_argument('--benchmark', type=str, default='coco', choices=['pascal', 'coco', 'fss'])
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--bsz', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--nworker', type=int, default=8)
    parser.add_argument('--seed', type=int, default=321)
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--condition', type=str, default='scribble', choices=['point', 'scribble', 'box', 'mask'])
    parser.add_argument('--use_ignore', type=bool, default=True, help='Boundaries are not considered during pascal training')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--num_query', type=int, default=50)
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['vgg16', 'resnet50', 'resnet101'])
    parser.add_argument('--save_path', type=str, default='model_checkpoint.pth', help='Path to save the model checkpoint')
    parser.add_argument('--load_path', type=str, default='', help='Path to load a pre-trained model checkpoint')
    args = parser.parse_args()

    # Distributed setting
    print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))
    print("Available GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    local_rank = args.local_rank
    dist.init_process_group(backend='nccl')
    print('local_rank: ', local_rank)
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

    if utils.is_main_process():
        Logger.initialize(args, training=True)
    utils.fix_randseed(args.seed)

    # Model initialization
    model = VRP_encoder(args, args.backbone, False)
    if utils.is_main_process():
        Logger.log_params(model)

    sam_model = SAM_pred()
    sam_model.to(device)
    model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                      output_device=args.local_rank, find_unused_parameters=True)

    optimizer = optim.AdamW([
        {'params': model.module.transformer_decoder.parameters()},
        {'params': model.module.downsample_query.parameters(), "lr": args.lr},
        {'params': model.module.merge_1.parameters(), "lr": args.lr},
    ], lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))

    # Dataset initialization
    FSSDataset.initialize(img_size=512, datapath=args.datapath, use_original_imgsize=False)
    dataloader_trn = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'trn')
    dataloader_val = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'val')

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(dataloader_trn))

    # Load model from checkpoint if specified
    if args.load_path and os.path.exists(args.load_path):
        checkpoint = torch.load(args.load_path, map_location=device)

        # Ensure 'model_state_dict' exists in the checkpoint
        if 'model_state_dict' not in checkpoint:
            raise RuntimeError(f"Missing 'model_state_dict' in checkpoint at {args.load_path}")
        model.load_state_dict(checkpoint['model_state_dict'])

        # Ensure 'optimizer_state_dict' exists in the checkpoint
        if 'optimizer_state_dict' not in checkpoint:
            raise RuntimeError(f"Missing 'optimizer_state_dict' in checkpoint at {args.load_path}")
        optimizer_state = checkpoint['optimizer_state_dict']
        optimizer.load_state_dict(optimizer_state)

        # Ensure 'scheduler_state_dict' exists in the checkpoint
        if 'scheduler_state_dict' not in checkpoint:
            raise RuntimeError(f"Missing 'scheduler_state_dict' in checkpoint at {args.load_path}")
        scheduler_state = checkpoint['scheduler_state_dict']
        scheduler.load_state_dict(scheduler_state)  # Restore scheduler state

        # Ensure 'epoch' exists in the checkpoint
        if 'epoch' not in checkpoint:
            raise RuntimeError(f"Missing 'epoch' in checkpoint at {args.load_path}")
        start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch

        # Ensure 'best_val_miou' exists in the checkpoint
        if 'best_val_miou' not in checkpoint:
            raise RuntimeError(f"Missing 'best_val_miou' in checkpoint at {args.load_path}")
        best_val_miou = checkpoint['best_val_miou']

        print(f"Loaded model from {args.load_path}, starting from epoch {start_epoch}")
    else:
        start_epoch = 0
        best_val_miou = float('-inf')
        print("No checkpoint found, training from scratch")
    '''
       else:
           raise RuntimeError(f"Checkpoint path '{args.load_path}' not found or is invalid")
    '''

    # Save the initial model state (including scheduler)
    if utils.is_main_process():
        torch.save({
            'epoch': start_epoch - 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),  # Save scheduler state (None initially)
            'best_val_miou': best_val_miou,
        }, args.save_path)
        print(f"Initial model state saved to {args.save_path}")

    # Freeze layers (optional)
    for param in model.module.layer0.parameters():
        param.requires_grad = False
    for param in model.module.layer1.parameters():
        param.requires_grad = False
    for param in model.module.layer2.parameters():
        param.requires_grad = False
    for param in model.module.layer3.parameters():
        param.requires_grad = False
    for param in model.module.layer4.parameters():
        param.requires_grad = False

    Evaluator.initialize(args)


    # Training Loop
    for epoch in range(start_epoch, args.epochs):
        # Training
        trn_loss, trn_miou, trn_fb_iou = train(args, epoch, model, sam_model, dataloader_trn, optimizer, scheduler,
                                               training=True)
        # Validation
        with torch.no_grad():
            val_loss, val_miou, val_fb_iou = train(args, epoch, model, sam_model, dataloader_val, optimizer, scheduler,
                                                   training=False)

        # Save the best model
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            if utils.is_main_process():
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),  # Save scheduler state
                    'best_val_miou': best_val_miou,
                }, args.save_path)
                print(f"Best model updated and saved to {args.save_path}")

        # Log to TensorBoard (optional)
        if utils.is_main_process():
            Logger.tbd_writer.add_scalars('data/loss', {'trn_loss': trn_loss, 'val_loss': val_loss}, epoch)
            Logger.tbd_writer.add_scalars('data/miou', {'trn_miou': trn_miou, 'val_miou': val_miou}, epoch)
            Logger.tbd_writer.add_scalars('data/fb_iou', {'trn_fb_iou': trn_fb_iou, 'val_fb_iou': val_fb_iou}, epoch)
            Logger.tbd_writer.flush()

    Logger.tbd_writer.close()
    Logger.info('==================== Finished Training ====================')
