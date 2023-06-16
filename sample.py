import warnings
import pandas as pd
import os
import random
import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings("ignore")


class CFG:
    comp_name = 'vesuvius'
    comp_dir_path = './'  # desired dir to store the data
    comp_folder_name = 'data' # desired folder name to store the data
    comp_dataset_path = f'{comp_dir_path}{comp_folder_name}/'
    slices = (20, 36)
    in_chans = slices[1] - slices[0]  # 65
    size = 224
    tile_size = 224
    sampling_method = "adaptive_stride_random"
    filter_method = "mean"
    stride = tile_size // 4
    neg_sample_size = 8


def read_image_mask(fragment_id):
    images = []
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


def adaptive(padded_mask, padded_volume, size):
    x_pos_list = []
    y_pos_list = []
    sampled_vols = []
    sampled_masks = []

    for y in range(0, padded_mask.shape[0], size[0]):
        for x in range(0, padded_mask.shape[1], size[0]):
            temp = padded_mask[y:y+size[0], x:x+size[1]]
            if temp.max() > 0:
                x_pos_list.append(x)
                y_pos_list.append(y)
                sampled_masks.append(temp)

    for i, (x, y) in tqdm(enumerate(zip(x_pos_list, y_pos_list)), total=len(x_pos_list), disable=True):
        temp_vol = []
        for img in padded_volume:
            image_roi = img[y:y + size[0], x:x + size[1]]
            temp_vol.append(image_roi)
        sampled_vols.append(temp_vol)

    return sampled_vols, sampled_masks


def random_sample(volume, mask, size=(256, 256)):
    h, w = mask.shape
    x, y = random.randint(0, w-size[1]), random.randint(0, h-size[0])
    sampled_mask = mask[y:y+size[0], x:x+size[1]]
    sampled_vol = [s[y:y+size[0], x:x+size[1]] for s in volume]
    while (np.max(sampled_mask) != 0) or (np.max(sampled_vol[0]) == 0):
        x, y = random.randint(0, w - size[1]), random.randint(0, h - size[0])
        sampled_mask = mask[y:y + size[0], x:x + size[1]]
        sampled_vol = [s[y:y+size[0], x:x+size[1]] for s in volume]
    return sampled_vol, sampled_mask


def random_n(n, volume, mask, size):
    sampled_vols = []
    sampled_masks = []

    for _ in tqdm(range(n), disable=True):
        sampled_vol, sampled_mask = random_sample(volume, mask, size)
        sampled_vols.append(sampled_vol)
        sampled_masks.append(sampled_mask)

    return sampled_vols, sampled_masks


def stride_moving_window(padded_mask, padded_volume, size, stride):
    x1_list = list(range(0, padded_mask.shape[1] - size[1] + 1, stride))
    y1_list = list(range(0, padded_mask.shape[0] - size[0] + 1, stride))
    sampled_vols = []
    sampled_masks = []
    for y in tqdm(y1_list, disable=True):
        for x in x1_list:
            temp = padded_mask[y:y + size[0], x:x + size[1]]
            sampled_masks.append(temp)
            temp_vol = []
            for img in padded_volume:
                image_roi = img[y:y + size[0], x:x + size[1]]
                temp_vol.append(image_roi)
            sampled_vols.append(temp_vol)
    return sampled_vols, sampled_masks


def store_train_images():
    root = f"{CFG.comp_dir_path}/{CFG.comp_folder_name}/sampled/{CFG.size}*{CFG.size}_stride{CFG.stride}_negsample{CFG.neg_sample_size}__slice{CFG.slices}method({CFG.sampling_method})/"
    os.makedirs(root, exist_ok=True)
    print("Loading...")
    table = {"v_path": [], "m_path": [], "fold": []}
    for fragment_id in range(1, 6):
        root_2 = root + f"{fragment_id}/"
        os.makedirs(root_2, exist_ok=True)
        m_root = root_2 + "masks/"
        v_root = root_2 + "volumes/"
        os.makedirs(m_root, exist_ok=True)
        os.makedirs(v_root, exist_ok=True)
        image, mask, ori_size = read_image_mask(fragment_id)
        size = (CFG.tile_size, CFG.tile_size)
        image = np.transpose(image, (2, 0, 1))

        if CFG.sampling_method == "adaptive_stride_random":
            sampled_vols_ad, sampled_masks_ad = adaptive(mask, image, size=(CFG.tile_size*4, CFG.tile_size*4))
            sampled_vols = []
            sampled_masks = []
            for _, (v, m) in tqdm(enumerate(zip(sampled_vols_ad, sampled_masks_ad)), total=len(sampled_masks_ad), disable=False):
                sampled_vols_s, sampled_masks_s = stride_moving_window(m, v, size, CFG.stride)
                sampled_vols_rand, sampled_masks_rand = random_n(CFG.neg_sample_size, image, mask, size)
                sampled_vols.extend(sampled_vols_rand)
                sampled_masks.extend(sampled_masks_rand)
                sampled_vols.extend(sampled_vols_s)
                sampled_masks.extend(sampled_masks_s)

        elif CFG.sampling_method == "stride":
            sampled_vols, sampled_masks = stride_moving_window(mask, image, size, CFG.stride)

        print(f"Saving {fragment_id}...")
        for j, (v, m) in tqdm(enumerate(zip(sampled_vols, sampled_masks)), total=len(sampled_masks), disable=False):
            m_pth = m_root + f"{j}.png"
            v_root_2 = v_root + f"{j}/"
            os.makedirs(v_root_2, exist_ok=True)
            cv2.imwrite(m_pth, m)
            for z, p in enumerate(v):
                v_path = v_root_2 + f"{z}.png"
                cv2.imwrite(v_path, p)
            table["v_path"].append(v_root_2)
            table["m_path"].append(m_pth)
            table["fold"].append(fragment_id)
    df = pd.DataFrame.from_dict(table)
    df.to_csv(root + "/train.csv", index=False)

if __name__ == "__main__":
    store_train_images()
