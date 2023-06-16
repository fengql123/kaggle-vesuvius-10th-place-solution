import numpy as np
import torch


def fbeta_numpy(targets, preds, beta=0.5, smooth=1e-5):
    """
    https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/397288
    """
    y_true_count = targets.sum()
    ctp = preds[targets == 1].sum()
    cfp = preds[targets == 0].sum()
    beta_squared = beta * beta

    c_precision = ctp / (ctp + cfp + smooth)
    c_recall = ctp / (y_true_count + smooth)
    dice = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall + smooth)

    return dice


def dice_coef_torch(targets, preds, beta=0.5, smooth=1e-5):

    # flatten label and prediction tensors
    preds = preds.contiguous().view(-1).float()
    targets = targets.contiguous().view(-1).float()

    y_true_count = targets.sum()
    ctp = preds[targets==1].sum()
    cfp = preds[targets==0].sum()
    beta_squared = beta * beta

    c_precision = ctp / (ctp + cfp + smooth)
    c_recall = ctp / (y_true_count + smooth)
    dice = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall + smooth)

    return dice


def calc_fbeta(mask, mask_pred, th):
    dice = dice_coef_torch(mask.long(), (mask_pred >= th).long(), beta=0.5)

    return dice


def calc_cv(mask_gts, mask_preds, orig_sizes, logger):
    best_th = 0
    best_dice = 0
    ths = np.array(range(30, 70 + 1, 5)) / 100
    orig_h, orig_w = orig_sizes
    mask_gt = mask_gts[:orig_h, :orig_w]
    mask_pred = mask_preds[:orig_h, :orig_w]
    #mask_pred = torch.sqrt(mask_pred)
    mask_gt = torch.from_numpy(mask_gt).to("cuda")

    for th in ths:
        dice = calc_fbeta(mask_gt, mask_pred, th)
        print(f"th: {th} dice: {dice}")

        if dice > best_dice:
            best_dice = dice
            best_th = th

    logger.info(f'best_th: {best_th}, fbeta: {best_dice}')

    return best_dice, best_th
