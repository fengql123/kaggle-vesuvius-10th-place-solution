from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import gc
import time
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
from torch.optim import AdamW
from lion_pytorch import Lion
import segmentation_models_pytorch as smp
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from warmup_scheduler import GradualWarmupScheduler
from config import CFG
from utils import AverageMeter, get_valid_images, get_mask_gt, get_ink_data
from dataset import TrainDataset, ValidDataset, get_transforms, InkDataset
from model import build_model
from validation import valid_fn, ink_valid_fn
from metrics import calc_cv
import matplotlib.pyplot as plt


class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    """
    https://www.kaggle.com/code/underwearfitting/single-fold-training-of-resnet200d-lb0-965
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(
            optimizer, multiplier, total_epoch, after_scheduler)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                    self.base_lrs]


def get_scheduler(cfg, optimizer, num_steps_total):
    if cfg.scheduler == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(cfg.num_warmup_steps_rate * num_steps_total),
            num_training_steps=num_steps_total
        )
    elif cfg.scheduler == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(cfg.num_warmup_steps_rate * num_steps_total),
            num_training_steps=num_steps_total, num_cycles=cfg.num_cycles
        )

    elif cfg.scheduler == 'gradualwarmup':
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            cfg.epochs,
            eta_min=1e-7
        )
        scheduler = GradualWarmupSchedulerV2(
            optimizer,
            multiplier=10,
            total_epoch=1,
            after_scheduler=scheduler_cosine
        )

    return scheduler


DiceLoss = smp.losses.DiceLoss(mode='binary')
BCELoss = smp.losses.SoftBCEWithLogitsLoss()

alpha = 0.5
beta = 1 - alpha
TverskyLoss = smp.losses.TverskyLoss(mode='binary', log_loss=False, alpha=alpha, beta=beta)
JaccardLoss = smp.losses.JaccardLoss(mode='binary', log_loss=False)
FocalLoss = smp.losses.FocalLoss(mode='binary')
LovaszLoss = smp.losses.LovaszLoss(mode='binary')

def criterion(y_pred, y_true):
    if CFG.loss == "DiceBCE":
        return 0.5 * BCELoss(y_pred, y_true) + 0.5 * DiceLoss(y_pred, y_true)
    elif CFG.loss == "BCE":
        return BCELoss(y_pred, y_true)
    elif CFG.loss == "Dice":
        return DiceLoss(y_pred, y_true)
    elif CFG.loss == "Tversky":
        return TverskyLoss(y_pred, y_true)
    elif CFG.loss == "BCETversky":
        return 0.5 * BCELoss(y_pred, y_true) + 0.5 * TverskyLoss(y_pred, y_true)
    elif CFG.loss == "mix":
        return torch.mean(BCELoss(y_pred, y_true) + TverskyLoss(y_pred, y_true) + JaccardLoss(y_pred, y_true) + FocalLoss(y_pred, y_true))

def train_fn(train_loader, model, criterion, optimizer, scheduler, device):
    model.train()

    scaler = GradScaler(enabled=CFG.use_amp)
    losses = AverageMeter()

    for step, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        with autocast(CFG.use_amp):
            y_preds = model(images)
            loss = criterion(y_preds, labels)

        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()

        if CFG.grad_clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        if CFG.scheduler != "gradualwarmup":
            scheduler.step()

    return losses.avg


def ink_classifier_train_fold(fold, logger, device):
    logger.info(f'----------------- Fold: {fold} -----------------')
    train_volumes, train_labels, validation_volumes, validation_labels, validation_pixels, validation_mask, validation_label = get_ink_data(fold, CFG.region_size, train_stride=CFG.region_stride, validation_stride=CFG.region_stride//2)
    train_dataset = InkDataset(train_volumes, train_labels, get_transforms(data='train', cfg=CFG))
    validation_dataset = InkDataset(validation_volumes, validation_labels, get_transforms(data='valid', cfg=CFG))
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.train_batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        validation_dataset,
        batch_size=CFG.valid_batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = build_model(CFG, logger)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    num_steps_total = CFG.epochs * len(train_loader)
    scheduler = get_scheduler(CFG, optimizer, num_steps_total)

    best_score = -1
    best_loss = np.inf
    best_pred = None

    for epoch in range(CFG.epochs):
        start_time = time.time()
        avg_loss = train_fn(train_loader, model, criterion, optimizer, scheduler, device)
        avg_val_loss, mask_preds = ink_valid_fn(valid_loader, model, criterion, device, validation_pixels, validation_mask)
        if CFG.scheduler == "gradualwarmup":
            scheduler.step()
        best_dice, best_th = calc_cv(validation_label, mask_preds, validation_label.shape, logger)
        score = best_dice
        elapsed = time.time() - start_time
        logger.info(
            f'Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        logger.info(
            f'Epoch {epoch + 1} - avgScore: {score:.4f}')
        update_best = score > best_score
        if update_best:
            best_loss = avg_val_loss
            best_score = score
            best_pred = mask_preds
            th = best_th
            logger.info(
                f'Epoch {epoch + 1} - Save Best Score: {best_score:.4f} Model')
            logger.info(
                f'Epoch {epoch + 1} - Save Best Loss: {best_loss:.4f} Model')
            torch.save(
                {
                    'model': model.state_dict(),
                    'encoder': model.encoder.state_dict(),
                    'pred': best_pred,
                }, CFG.model_dir + f'{CFG.model}_fold{fold}_best.pth')
    gc.collect()
    return best_score, th


def segmentation_model_train_fold(fold, logger, device):
    logger.info(f'----------------- Fold: {fold} -----------------')

    valid_image, valid_mask, valid_xyxys, valid_size = get_valid_images(fold)
    train_df = pd.read_csv(CFG.train_dir + "train.csv")
    train_df = train_df[train_df.fold != fold]
    train_dataset = TrainDataset(train_df, transform=get_transforms(data='train', cfg=CFG))
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.train_batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_dataset = ValidDataset(
        valid_image,
        CFG,
        labels=valid_mask,
        transform=get_transforms(data='valid', cfg=CFG)
        )
    valid_loaders = DataLoader(
        valid_dataset,
        batch_size=CFG.valid_batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False
    )

    valid_xyxys = np.stack(valid_xyxys)
    valid_mask_gts = get_mask_gt(fold)

    model = build_model(CFG, logger)
    if CFG.use_pretrained:
        state_dict = torch.load(CFG.pretrained_dir + f"ink-classifier_fold{fold}_best.pth")["encoder"]
        model.encoder.load_state_dict(state_dict, strict=True)
        
    if CFG.use_pretrained_model:
        state_dict = torch.load(CFG.pretrained_dir + f"3D-2D_fold{fold}_best.pth")["model"]
        model.load_state_dict(state_dict, strict=True)
    
    if CFG.freeze:
        for param in model.encoder.parameters():
            param.requires_grad = False
        
    if CFG.llrd:
        optimizer_params = [
            {"params": model.encoder.parameters(), "lr": CFG.lr/20},
            {"params": model.decoder.parameters(), "lr": CFG.lr}
        ]
        if CFG.optimizer == "AdamW":
            optimizer = AdamW(optimizer_params, lr=CFG.lr, weight_decay=CFG.weight_decay)
        elif CFG.optimizer == "Lion":
            optimizer = Lion(optimizer_params, lr=CFG.lr, weight_decay=CFG.weight_decay)
    else:
        if CFG.optimizer == "AdamW":
            optimizer = AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
        elif CFG.optimizer == "Lion":
            optimizer = Lion(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
        
    model.to(device)
    num_steps_total = CFG.epochs * len(train_loader)
    scheduler = get_scheduler(CFG, optimizer, num_steps_total)

    best_score = -1
    best_loss = np.inf
    best_pred = None

    for epoch in range(CFG.epochs):
        start_time = time.time()
        avg_loss = train_fn(train_loader, model, criterion, optimizer, scheduler, device)
        avg_val_loss, mask_preds = valid_fn(valid_loaders, model, criterion, device, valid_xyxys, valid_mask_gts)
        if CFG.scheduler == "gradualwarmup":
            scheduler.step()
        best_dice, best_th = calc_cv(valid_mask_gts, mask_preds, valid_size, logger)
        score = best_dice
        elapsed = time.time() - start_time
        logger.info(
            f'Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        logger.info(
            f'Epoch {epoch + 1} - avgScore: {score:.4f}')
        update_best = score > best_score

        if update_best:
            best_loss = avg_val_loss
            best_score = score
            best_pred = mask_preds
            th = best_th
            logger.info(
                f'Epoch {epoch + 1} - Save Best Score: {best_score:.4f} Model')
            logger.info(
                f'Epoch {epoch + 1} - Save Best Loss: {best_loss:.4f} Model')
            torch.save(
                {
                    'model': model.state_dict(),
                    'encoder': model.encoder.state_dict(),
                    'pred': best_pred,
                }, CFG.model_dir + f'{CFG.model}_fold{fold}_best.pth')
    gc.collect()
    return best_score, th
