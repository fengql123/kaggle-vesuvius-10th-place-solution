from torch.utils.data import Dataset
import glob
import cv2
import numpy as np
import torch
import albumentations as A
from config import CFG


def get_transforms(data, cfg):
    if data == 'train':
        aug = A.Compose(cfg.train_aug_list)
    elif data == 'valid':
        aug = A.Compose(cfg.valid_aug_list)
    return aug


class TrainDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __getitem__(self, idx):
        v_root = self.df.iloc[idx]["v_path"]
        v_paths = glob.glob(v_root + "*.png")
        image = np.stack([cv2.imread(v_root + f"{i}.png", 0) for i in range(len(v_paths))], axis=-1)
        m_path = self.df.iloc[idx]["m_path"]
        label = cv2.imread(m_path, 0)
        label = np.expand_dims(label, axis=-1)
        label = label.astype('float32')
        label /= 255.0
        if self.transform:
            data = self.transform(image=image, mask=label)
            image = data['image']
            label = data['mask']
        if CFG.model != "2.5D":
            image = torch.unsqueeze(image, dim=0)
        return image, label

    def __len__(self):
        return len(self.df)


class ValidDataset(Dataset):
    def __init__(self, images, cfg, labels=None, transform=None):
        self.images = images
        self.cfg = cfg
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            data = self.transform(image=image, mask=label)
            image = data['image']
            label = data['mask']

        if CFG.model != "2.5D":
            image = torch.unsqueeze(image, dim=0)

        return image, label


class InkDataset(Dataset):
    def __init__(self, volumes, labels, transform=None, mode="train"):
        self.volumes = volumes
        if mode == "train":
            self.labels = torch.Tensor(labels).float()
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.volumes)

    def __getitem__(self, idx):
        image = self.volumes[idx]
        if self.transform:
            data = self.transform(image=image)
            image = data['image']
        image = torch.unsqueeze(image, dim=0)
        if self.mode == "train":
            label = self.labels[idx]
            label = torch.unsqueeze(label, dim=0)
            return image, label
        else:
            return image


