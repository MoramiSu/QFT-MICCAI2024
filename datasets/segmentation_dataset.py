import pickle

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
from albumentations import Compose, Normalize, Resize, ShiftScaleRotate
from albumentations.pytorch import ToTensorV2
from constants import *
from datasets.classification_dataset import BaseImageDataset
from datasets.utils import resize_img, get_imgs
from PIL import Image

np.random.seed(42)

class BUSISegmentDataset(BaseImageDataset):
    def __init__(self, split="train", transform=None, data_pct=1., imsize=224) -> None:
        super().__init__(split, transform)

        if not os.path.exists(BUSI_DATA_DIR):
            raise RuntimeError(f"{BUSI_DATA_DIR} does not exist!")

        if self.split == "train":
            with open(BUSI_SEG_TRAIN_PKL, "rb") as f:
                self.filenames, self.bboxs = pickle.load(f)
        elif self.split == "valid":
            with open(BUSI_SEG_VALID_PKL, "rb") as f:
                self.filenames, self.bboxs = pickle.load(f)
        elif self.split == "test":
            with open(BUSI_SEG_TEST_PKL, "rb") as f:
                self.filenames, self.bboxs = pickle.load(f)
        else:
            raise ValueError(f"split {split} does not exist!")


        n = len(self.filenames)
        if split == "train":
            indices = np.random.choice(n, int(data_pct * n), replace=False)
            self.filenames = self.filenames[indices]
            self.bboxs = self.bboxs[indices]

        self.imsize = imsize
        self.seg_transform = self.get_transforms()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        box = self.bboxs[index]
        img_path = BUSI_IMG_DIR / filename
        x = cv2.imread(str(img_path), 0)
        x = cv2.convertScaleAbs(x, alpha=(255.0 / x.max()))
        x = np.asarray(Image.fromarray(x).convert('RGB'))
        mask_path = BUSI_IMG_DIR / filename.replace('.png', '_mask.png')
        mask = cv2.imread(str(mask_path), 0)
        if len(box) > 1:
            for i in range(1, len(box)):
                mask_path = BUSI_IMG_DIR / filename.replace('.png', '_mask_'+str(i)+'.png')
                mask += cv2.imread(str(mask_path), 0)
        mask = (mask >= 1).astype("float32")
        mask = resize_img(mask, self.imsize)
        augmented = self.seg_transform(image=x, mask=mask)

        x = augmented["image"]
        y = augmented["mask"]

        return {
            "imgs": x,
            "labels": y,
            "filenames": filename
        }

    def get_transforms(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        list_transforms = []
        if self.split == "train":
            list_transforms.extend(
                [
                    ShiftScaleRotate(
                        shift_limit=0,
                        scale_limit=0.1,
                        rotate_limit=10,
                        p=0.5,
                        border_mode=cv2.BORDER_CONSTANT,
                    )
                ]
            )
        list_transforms.extend(
            [
                Resize(self.imsize, self.imsize),
                Normalize(mean=mean, std=std, p=1),
                ToTensorV2(),
            ]
        )

        list_trfms = Compose(list_transforms)
        return list_trfms

class DDTISegmentDataset(BaseImageDataset):
    def __init__(self, split="train", transform=None, data_pct=1., imsize=224) -> None:
        super().__init__(split, transform)

        if not os.path.exists(DDTI_DATA_DIR):
            raise RuntimeError(f"{DDTI_DATA_DIR} does not exist!")

        if self.split == "train":
            with open(DDTI_SEG_TRAIN_PKL, "rb") as f:
                self.filenames, self.bboxs = pickle.load(f)
        elif self.split == "valid":
            with open(DDTI_SEG_VALID_PKL, "rb") as f:
                self.filenames, self.bboxs = pickle.load(f)
        elif self.split == "test":
            with open(DDTI_SEG_TEST_PKL, "rb") as f:
                self.filenames, self.bboxs = pickle.load(f)
        else:
            raise ValueError(f"split {split} does not exist!")

        n = len(self.filenames)
        if split == "train":
            indices = np.random.choice(n, int(data_pct * n), replace=False)
            self.filenames = self.filenames[indices]
            self.bboxs = self.bboxs[indices]

        self.imsize = imsize
        self.seg_transform = self.get_transforms()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        box = self.bboxs[index]
        img_path = DDTI_IMG_DIR / filename
        x = cv2.imread(str(img_path), 0)
        x = cv2.convertScaleAbs(x, alpha=(255.0 / x.max()))
        x = np.asarray(Image.fromarray(x).convert('RGB'))
        mask_path = DDTI_IMG_DIR / filename.replace('.PNG', '_mask.PNG')
        mask = cv2.imread(str(mask_path), 0)
        # if len(box) > 1:
            # for i in range(1, len(box)):
                # mask_path = DDTI_IMG_DIR / filename.replace('.png', '_mask_'+str(i)+'.png')
                # mask += cv2.imread(str(mask_path), 0)
        mask = (mask >= 1).astype("float32")
        mask = resize_img(mask, self.imsize)
        augmented = self.seg_transform(image=x, mask=mask)

        x = augmented["image"]
        y = augmented["mask"]

        return {
                "imgs": x,
                "labels": y,
                "filenames": filename
            }

    def get_transforms(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        list_transforms = []
        if self.split == "train":
            list_transforms.extend(
                [
                    ShiftScaleRotate(
                        shift_limit=0,
                        scale_limit=0.1,
                        rotate_limit=10,
                        p=0.5,
                        border_mode=cv2.BORDER_CONSTANT,
                    )
                ]
            )
        list_transforms.extend(
            [
                Resize(self.imsize, self.imsize),
                Normalize(mean=mean, std=std, p=1),
                ToTensorV2(),
            ]
        )

        list_trfms = Compose(list_transforms)
        return list_trfms

def seg_collate_fn(batch):
    """sort sequence"""
    imgs, masks, filenames = [], [], []
    for b in batch:
        img, mask, filename = b
        imgs.append(img)
        masks.append(mask)
        filenames.append(filename)

    # stack
    imgs = torch.stack(imgs)
    masks = torch.stack(masks)

    return_dict = {
        "imgs": imgs,
        'masks': masks,
        'filenames': filenames
    }
    return return_dict

if __name__ == "__main__":
    dataset = BUSISegmentDataset()
    for data in dataset:
        pass
