import numpy as np
from tqdm.auto import tqdm
import torch
from utils import AverageMeter
from config import CFG


def valid_fn(valid_loaders, model, criterion, device, valid_xyxys, valid_mask_gts):
    model.eval()
    losses = AverageMeter()

    mask_pred = torch.zeros(valid_mask_gts.shape).to('cuda')
    mask_count = torch.zeros(valid_mask_gts.shape).to('cuda')

    for step, (images, labels) in tqdm(enumerate(valid_loaders), total=len(valid_loaders)):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.no_grad():
            if CFG.tta:
                xs = [
                    images,
                    torch.rot90(images, k=1, dims=(-2, -1)),
                    torch.rot90(images, k=2, dims=(-2, -1)),
                    torch.rot90(images, k=3, dims=(-2, -1)),
                ]
                temp = []
                for x in xs:
                    temp.append(model(x))
                outputs = [
                    temp[0],
                    torch.rot90(temp[1], k=-1, dims=(-2, -1)),
                    torch.rot90(temp[2], k=-2, dims=(-2, -1)),
                    torch.rot90(temp[3], k=-3, dims=(-2, -1)),
                ]
                y_preds = torch.mean(torch.stack(outputs), dim=0)
            else:
                y_preds = model(images)

            loss = criterion(y_preds, labels)
        losses.update(loss.item(), batch_size)

        # make whole mask
        y_preds = torch.sigmoid(y_preds)
        start_idx = step * CFG.valid_batch_size
        end_idx = start_idx + batch_size
        for i, (x1, y1, x2, y2) in enumerate(valid_xyxys[start_idx:end_idx]):
            mask_pred[y1:y2, x1:x2] += y_preds[i].squeeze(0)
            mask_count[y1:y2, x1:x2] += torch.ones((CFG.tile_size, CFG.tile_size)).to('cuda')

    mask_pred /= mask_count

    return losses.avg, mask_pred


def ink_valid_fn(loader, model, criterion, device, pixels, mask):
    out = torch.zeros(mask.shape).to('cuda')
    mask_count = torch.zeros(mask.shape).to('cuda')
    mask_count += torch.from_numpy((1 - mask)).to('cuda')
    mask = torch.from_numpy(mask).to('cuda')
    model.eval()
    radius = CFG.region_size//2
    losses = AverageMeter()
    for i, (images, labels) in tqdm(enumerate(loader), total=len(loader)):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.no_grad():
            if CFG.tta:
                xs = [
                    images,
                    torch.rot90(images, k=1, dims=(-2, -1)),
                    torch.rot90(images, k=2, dims=(-2, -1)),
                    torch.rot90(images, k=3, dims=(-2, -1)),
                ]
                temp = []
                for x in xs:
                    temp.append(model(x))
                preds = torch.mean(torch.stack(temp), dim=0)
            else:
                preds = model(images)
            loss = criterion(preds, labels)
            losses.update(loss.item(), batch_size)
            preds = torch.sigmoid(preds)
            for j, value in enumerate(preds):
                y, x = pixels[(i * batch_size) + j]
                out[y-radius:y+radius, x-radius: x+radius] += value
                mask_count[y-radius:y+radius, x-radius: x+radius] += torch.ones((CFG.region_size, CFG.region_size)).to('cuda')

    out /= mask_count
    out *= mask
    return losses.avg, out


def ink_infer_fn(loader, model, device, pixels, mask):
    out = np.zeros_like(mask).astype("float")
    mask_count = np.zeros_like(mask)
    mask_count += (1 - mask)
    model.to(device)
    model.eval()
    radius = CFG.region_size//2
    for i, (images) in tqdm(enumerate(loader), total=len(loader)):
        images = images.to(device)
        batch_size = images.size(0)
        with torch.no_grad():
            preds = model(images)
            preds = torch.sigmoid(preds).to('cpu').numpy()
            for j, value in enumerate(preds):
                y, x = pixels[(i * batch_size) + j]
                out[y-radius:y+radius, x-radius: x+radius] += value
                mask_count[y-radius:y+radius, x-radius: x+radius] += np.ones((CFG.region_size, CFG.region_size))

    out /= mask_count
    out *= mask
    return out
