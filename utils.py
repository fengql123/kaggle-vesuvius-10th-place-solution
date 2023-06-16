import os
import random
import cv2
from PIL import Image
import numpy as np
from tqdm.auto import tqdm
import torch
from config import CFG


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def init_logger(log_file):
    from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def set_seed(seed=None, cudnn_deterministic=True):
    if seed is None:
        seed = 42

    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = False


def make_dirs(cfg):
    for dir in [cfg.model_dir, cfg.log_dir]:
        os.makedirs(dir, exist_ok=True)


def cfg_init(cfg, mode='train'):
    # set_env_name()
    # set_dataset_path(cfg)

    if mode == 'train':
        make_dirs(cfg)


def load_volume(volume_path, disable_tqdm=True):
    volume = []
    for i in tqdm(range(65), disable=disable_tqdm):
        img = Image.open(f"{volume_path}/{i:02}.tif")
        arr = np.array(img)
        volume.append(arr)
    return volume


def get_indices(fragment_id):
    vol = load_volume(CFG.comp_dataset_path + f"train/{fragment_id}/surface_volume")
    inklabels = np.array(Image.open(CFG.comp_dataset_path + f"train/{fragment_id}/inklabels.png"))
    mask = np.array(Image.open(CFG.comp_dataset_path + f"train/{fragment_id}/mask.png"))

    def get(v):
        return v[v > 0]

    labeled_mean = np.array([np.mean(get(v * inklabels)) for v in vol])
    unlabeled_mean = np.array([np.mean(get(v * (mask - inklabels))) for v in vol])
    labeled_median = np.array([np.median(get(v * inklabels)) for v in vol])
    unlabeled_median = np.array([np.median(get(v * (mask - inklabels))) for v in vol])
    diff_mean = np.absolute(unlabeled_mean - labeled_mean)
    diff_median = np.absolute(unlabeled_median - labeled_median)

    return np.argsort(diff_mean)[-CFG.in_chans:] if CFG.filter_method == "mean" else np.argsort(diff_median)[-CFG.in_chans:]


def read_image_mask(fragment_id):
    images = []

    # mid = 65 // 2
    # start = mid - CFG.in_chans // 2
    # start = 0
    # end = mid + CFG.in_chans // 2
    # end = CFG.in_chans
    # idxs = np.sort(get_indices(fragment_id))
    idxs = range(CFG.slices[0], CFG.slices[1])

    for i in idxs:
        image = cv2.imread(CFG.comp_dataset_path + f"train/{fragment_id}/surface_volume/{i:02}.tif", 0)

        pad0 = (CFG.tile_size - image.shape[0] % CFG.tile_size)
        pad1 = (CFG.tile_size - image.shape[1] % CFG.tile_size)

        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)

        images.append(image)
    images = np.stack(images, axis=2)

    mask = cv2.imread(CFG.comp_dataset_path + f"train/{fragment_id}/inklabels.png", 0)
    ori_h, ori_w = mask.shape
    mask = np.pad(mask, [(0, pad0), (0, pad1)], constant_values=0)

    return images, mask, (ori_h, ori_w)


def get_valid_images(valid_id):
    valid_images = []
    valid_masks = []
    valid_xyxys = []
    image, mask, ori_size = read_image_mask(valid_id)
    mask = mask.astype('float32')
    mask /= 255.0
    size = (CFG.tile_size, CFG.tile_size)
    x1_list = list(range(0, mask.shape[1] - size[1] + 1, CFG.valid_stride))
    y1_list = list(range(0, mask.shape[0] - size[0] + 1, CFG.valid_stride))
    for y1 in y1_list:
        for x1 in x1_list:
            y2 = y1 + CFG.tile_size
            x2 = x1 + CFG.tile_size
            # xyxys.append((x1, y1, x2, y2))
            valid_images.append(image[y1:y2, x1:x2])
            valid_masks.append(mask[y1:y2, x1:x2, None])
            valid_xyxys.append([x1, y1, x2, y2])

    return valid_images, valid_masks, valid_xyxys, ori_size


def get_ink_data_fold(fold, size, stride=0):
    mask_path = CFG.comp_dataset_path + f"train/{fold}/mask.png"
    label_path = CFG.comp_dataset_path + f"train/{fold}/inklabels.png"
    mask = cv2.imread(mask_path, 0) / 255.
    label = cv2.imread(label_path, 0) /255.
    images = []
    idxs = range(CFG.slices[0], CFG.slices[1])
    for i in idxs:
        image = cv2.imread(CFG.comp_dataset_path + f"train/{fold}/surface_volume/{i:02}.tif", 0)
        images.append(image)
    images = np.stack(images, axis=2)
    radius = int(size // 2)
    # Create a Boolean array mask of the same shape as the mask, initially all True
    not_border = np.zeros(mask.shape, dtype=bool)
    not_border[radius:mask.shape[0] - radius, radius:mask.shape[1] - radius] = True
    arr_mask = np.array(mask) * not_border
    if stride != 0:
        sparse_mask = np.zeros(mask.shape, dtype=bool)
        sparse_mask[::stride, ::stride] = True
        pixels = np.argwhere(sparse_mask * arr_mask)
    else:
        pixels = np.argwhere(arr_mask)
    return images, label, pixels, mask


def get_ink_data(fold, size, train_stride=0, validation_stride=0):
    train_volumes = []
    train_labels = []
    validation_volumes = []
    validation_labels = []
    validation_pixels = None
    validation_mask = None
    validation_label = None
    radius = int(size//2)
    for i in range(1, 6):
        if i != fold:
            images, ink_labels, pixels, _ = get_ink_data_fold(i, size, train_stride)
        else:
            images, ink_labels, pixels, mask = get_ink_data_fold(i, size, validation_stride)
            validation_pixels = pixels
            validation_mask = mask
            validation_label = ink_labels
        for y, x in pixels:
            subvolume = images[y-radius:y+radius, x-radius:x+radius, :]
            label = ink_labels[y, x]
            if i != fold:
                train_volumes.append(subvolume)
                train_labels.append(label)
            else:
                validation_volumes.append(subvolume)
                validation_labels.append(label)
    return train_volumes, train_labels, validation_volumes, validation_labels, validation_pixels, validation_mask, validation_label


def get_ink_data_infer(dir, size, stride=0):
    mask_path = f"{dir}/mask.png"
    mask = cv2.imread(mask_path, 0)
    mask = mask / 255.0
    images = []
    idxs = range(CFG.slices[0], CFG.slices[1])
    for i in idxs:
        image = cv2.imread(f"{dir}/surface_volume/{i:02}.tif", 0)
        images.append(image)
    images = np.stack(images, axis=2)
    radius = int(size // 2)
    # Create a Boolean array mask of the same shape as the mask, initially all True
    not_border = np.zeros(mask.shape, dtype=bool)
    not_border[radius:mask.shape[0] - radius, radius:mask.shape[1] - radius] = True
    arr_mask = np.array(mask) * not_border
    if stride != 0:
        sparse_mask = np.zeros(mask.shape, dtype=bool)
        sparse_mask[::stride, ::stride] = True
        pixels = np.argwhere(sparse_mask * arr_mask)
    else:
        pixels = np.argwhere(arr_mask)

    volumes = []
    for y, x in pixels:
        subvolume = images[y - radius:y + radius, x - radius:x + radius, :]
        volumes.append(subvolume)
    return volumes, pixels, mask


def get_mask_gt(v_id):
    valid_mask_gt = cv2.imread(CFG.comp_dataset_path + f"train/{v_id}/inklabels.png", 0)
    valid_mask_gt = valid_mask_gt / 255
    pad0 = (CFG.tile_size - valid_mask_gt.shape[0] % CFG.tile_size)
    pad1 = (CFG.tile_size - valid_mask_gt.shape[1] % CFG.tile_size)
    valid_mask_gt = np.pad(valid_mask_gt, [(0, pad0), (0, pad1)], constant_values=0)
    return valid_mask_gt
