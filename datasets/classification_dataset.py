import os
from datasets.transforms import DataTransforms
import numpy as np
import pandas as pd
import torch
from constants import *
from datasets.utils import get_imgs
from torch.utils.data import Dataset

np.random.seed(42)

class BaseImageDataset(Dataset):
    def __init__(self, split="train", transform=None) -> None:
        super().__init__()

        self.split = split
        self.transform = transform

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

class BUSIImageDataset(BaseImageDataset):
    def __init__(self, split="train", transform=None,
                 data_pct=0.01, imsize=256) -> None:
        super().__init__(split=split, transform=transform)

        if not os.path.exists(BUSI_DATA_DIR):
            raise RuntimeError(f"{BUSI_DATA_DIR} does not exist!")

        if self.split == "train":
            self.df = pd.read_csv(BUSI_CLASSIFICATION_TRAIN_CSV)
            self.df["filename"] = self.df["filename"].apply(
                lambda x: BUSI_DATA_DIR / f"files/{x}")
        elif self.split == "valid":
            self.df = pd.read_csv(BUSI_CLASSIFICATION_VALID_CSV)
            self.df["filename"] = self.df["filename"].apply(
                lambda x: BUSI_DATA_DIR / f"files/{x}")
        elif self.split == "test":
            self.df = pd.read_csv(BUSI_CLASSIFICATION_TEST_CSV)
            self.df["filename"] = self.df["filename"].apply(
                lambda x: BUSI_DATA_DIR / f"files/{x}")
        else:
            raise ValueError(f"split {split} does not exist!")

        if data_pct != 1 and self.split == "train":
            self.df = self.df.sample(frac=data_pct, random_state=42)

        self.imsize = imsize

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_path = row["filename"]
        x = get_imgs(img_path, self.imsize, self.transform)
        y = float(row["label"])
        y = torch.tensor([y])

        return x, y

class AUIDTImageDataset(BaseImageDataset):
    def __init__(self, split="train", transform=None,
                 data_pct=0.01, imsize=256) -> None:
        super().__init__(split=split, transform=transform)

        if not os.path.exists(AUIDT_DATA_DIR):
            raise RuntimeError(f"{AUIDT_DATA_DIR} does not exist!")

        if self.split == "train":
            self.df = pd.read_csv(AUIDT_TRAIN_CSV)
            self.df["filenames"] = self.df["filenames"].apply(
                lambda x: AUIDT_DATA_DIR / f"files/{x}")
        elif self.split == "valid":
            self.df = pd.read_csv(AUIDT_VALID_CSV)
            self.df["filenames"] = self.df["filenames"].apply(
                lambda x: AUIDT_DATA_DIR / f"files/{x}")
        elif self.split == "test":
            self.df = pd.read_csv(AUIDT_TEST_CSV)
            self.df["filenames"] = self.df["filenames"].apply(
                lambda x: AUIDT_DATA_DIR / f"files/{x}")
        else:
            raise ValueError(f"split {split} does not exist!")

        if data_pct != 1 and self.split == "train":
            self.df = self.df.sample(frac=data_pct, random_state=42)

        self.imsize = imsize

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        # get image
        img_path = row["filenames"]
        x = get_imgs(img_path, self.imsize, self.transform)
        y = float(row["labels"])
        y = torch.tensor([y])

        return x, y

if __name__ == '__main__':
    transform = DataTransforms()
    dataset = AUIDTImageDataset(split='test', transform=transform)
    for i in dataset:
        print(i[1])
