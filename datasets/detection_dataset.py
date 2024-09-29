import pickle
import random

from constants import *
from datasets.classification_dataset import BaseImageDataset
from datasets.transforms import *
from PIL import Image

random.seed(42)
np.random.seed(42)

class BUSIDetectionDataset(BaseImageDataset):
    def __init__(self, split="train", transform=None, data_pct=1., imsize=224, max_objects=10):
        super().__init__(split, transform)
        if not os.path.exists(BUSI_DATA_DIR):
            raise RuntimeError(f"{BUSI_DATA_DIR} does not exist!")

        if self.split == "train":
            with open(BUSI_DETECTION_TRAIN_PKL, "rb") as f:
                filenames, bboxs = pickle.load(f)
        elif self.split == "valid":
            with open(BUSI_DETECTION_VALID_PKL, "rb") as f:
                filenames, bboxs = pickle.load(f)
        elif self.split == "test":
            with open(BUSI_DETECTION_TEST_PKL, "rb") as f:
                filenames, bboxs = pickle.load(f)
        else:
            raise ValueError(f"split {split} does not exist!")

        self.imsize = imsize

        self.filenames_list, self.bboxs_list = [], []
        for i in range(len(filenames)):
            bbox = np.array(bboxs[i])
            new_bbox = bbox.copy()
            new_bbox[:, 0] = (bbox[:, 0] + bbox[:, 2]) / 2.
            new_bbox[:, 1] = (bbox[:, 1] + bbox[:, 3]) / 2.
            new_bbox[:, 2] = bbox[:, 2] - bbox[:, 0]
            new_bbox[:, 3] = bbox[:, 3] - bbox[:, 1]
            n = new_bbox.shape[0]
            new_bbox = np.hstack([np.zeros((n, 1)), new_bbox])
            pad = np.zeros((max_objects - n, 5))
            new_bbox = np.vstack([new_bbox, pad])
            self.filenames_list.append(filenames[i])
            self.bboxs_list.append(new_bbox)

        self.filenames_list = np.array(self.filenames_list)
        self.bboxs_list = np.array(self.bboxs_list)
        n = len(self.filenames_list)
        if split == "train":
            indices = np.random.choice(n, int(data_pct * n), replace=False)
            self.filenames_list = self.filenames_list[indices]
            self.bboxs_list = self.bboxs_list[indices]

    def __len__(self):
        return len(self.filenames_list)

    def __getitem__(self, index):
        filename = self.filenames_list[index]
        img_path = BUSI_DATA_DIR / ('files'+'/'+filename)
        x = cv2.imread(str(img_path), 0)
        x = cv2.convertScaleAbs(x, alpha=(255.0 / x.max()))
        h, w = x.shape
        x = cv2.resize(x, (self.imsize, self.imsize),
                       interpolation=cv2.INTER_LINEAR)
        img = Image.fromarray(x).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        y = self.bboxs_list[index]
        y[:, 1] /= w
        y[:, 3] /= w
        y[:, 2] /= h
        y[:, 4] /= h

        sample = {
            "imgs": img,
            "labels": y,
            "filenames": filename
        }

        return sample

class DDTIDetectionDataset(BaseImageDataset):
    def __init__(self, split="train", transform=None, data_pct=1., imsize=224, max_objects=10):
        super().__init__(split, transform)
        if not os.path.exists(DDTI_DATA_DIR):
            raise RuntimeError(f"{DDTI_DATA_DIR} does not exist!")

        if self.split == "train":
            with open(DDTI_DETECTION_TRAIN_PKL, "rb") as f:
                filenames, bboxs = pickle.load(f)
        elif self.split == "valid":
            with open(DDTI_DETECTION_VALID_PKL, "rb") as f:
                filenames, bboxs = pickle.load(f)
        elif self.split == "test":
            with open(DDTI_DETECTION_TEST_PKL, "rb") as f:
                filenames, bboxs = pickle.load(f)
        else:
            raise ValueError(f"split {split} does not exist!")

        self.imsize = imsize

        self.filenames_list, self.bboxs_list = [], []
        for i in range(len(filenames)):
            bbox = np.array(bboxs[i])
            new_bbox = bbox.copy()
            new_bbox[:, 0] = (bbox[:, 0] + bbox[:, 2]) / 2.
            new_bbox[:, 1] = (bbox[:, 1] + bbox[:, 3]) / 2.
            new_bbox[:, 2] = bbox[:, 2] - bbox[:, 0]
            new_bbox[:, 3] = bbox[:, 3] - bbox[:, 1]
            n = new_bbox.shape[0]
            new_bbox = np.hstack([np.zeros((n, 1)), new_bbox])
            pad = np.zeros((max_objects - n, 5))
            new_bbox = np.vstack([new_bbox, pad])
            self.filenames_list.append(filenames[i])
            self.bboxs_list.append(new_bbox)

        self.filenames_list = np.array(self.filenames_list)
        self.bboxs_list = np.array(self.bboxs_list)
        n = len(self.filenames_list)
        if split == "train":
            indices = np.random.choice(n, int(data_pct * n), replace=False)
            self.filenames_list = self.filenames_list[indices]
            self.bboxs_list = self.bboxs_list[indices]

    def __len__(self):
        return len(self.filenames_list)

    def __getitem__(self, index):
        filename = self.filenames_list[index]
        img_path = DDTI_DATA_DIR / ('files'+'/'+filename)
        x = cv2.imread(str(img_path), 0)
        x = cv2.convertScaleAbs(x, alpha=(255.0 / x.max()))
        h, w = x.shape
        x = cv2.resize(x, (self.imsize, self.imsize),
                       interpolation=cv2.INTER_LINEAR)
        img = Image.fromarray(x).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        y = self.bboxs_list[index]
        y[:, 1] /= w
        y[:, 3] /= w
        y[:, 2] /= h
        y[:, 4] /= h

        sample = {
            "imgs": img,
            "labels": y,
            "filenames": filename
        }

        return sample

if __name__ == "__main__":
    dataset = BUSIDetectionDataset(split='test')
    print(len(dataset))
    print(dataset[5])
