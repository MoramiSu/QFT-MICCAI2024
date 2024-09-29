import json

import torch
import torch.utils.data as data
from constants import *
from transformers import BertTokenizer
import cv2
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class MultimodalPretrainingDataset(data.Dataset):
    def __init__(self, split="train", transform=None, data_pct=1.0,
                 imsize=256, text0_max_words=200, text1_max_words=100, text2_max_words=50):
        super().__init__()
        if split == 'train':
            with open(ULTRA_COARSE_TRAIN, 'r', encoding='utf-8') as f:
                self.coarse_grain = json.load(f)
            if os.path.exists(ULTRA_MIDDLE_TRAIN):
                with open(ULTRA_MIDDLE_TRAIN, 'r', encoding='utf-8') as f:
                    self.middle_grain = json.load(f)
            else:
                self.middle_grain = None
            if os.path.exists(ULTRA_FINE_TRAIN):
                with open(ULTRA_FINE_TRAIN, 'r', encoding='utf-8') as f:
                    self.fine_grain = json.load(f)
            else:
                self.fine_grain = None
        else:
            with open(ULTRA_COARSE_VAL, 'r', encoding='utf-8') as f:
                self.coarse_grain = json.load(f)
            if os.path.exists(ULTRA_MIDDLE_VAL):
                with open(ULTRA_MIDDLE_VAL, 'r', encoding='utf-8') as f:
                    self.middle_grain = json.load(f)
            else:
                self.middle_grain = None
            if os.path.exists(ULTRA_FINE_VAL):
                with open(ULTRA_FINE_VAL, 'r', encoding='utf-8') as f:
                    self.fine_grain = json.load(f)
            else:
                self.fine_grain = None
        # elif split == 'test':
        #     with open(ULTRA_COARSE_TEST, 'r', encoding='utf-8') as f:
        #         self.coarse_grain = json.load(f)
        #     if os.path.exists(ULTRA_MIDDLE_VAL):
        #         with open(ULTRA_MIDDLE_VAL, 'r', encoding='utf-8') as f:
        #             self.middle_grain = json.load(f)
        #     else:
        #         self.middle_grain = None
        #     if os.path.exists(ULTRA_FINE):
        #         with open(ULTRA_FINE, 'r', encoding='utf-8') as f:
        #             self.fine_grain = json.load(f)
        #     else:
        #         self.fine_grain = None

        self.data_idx = []
        for key in self.coarse_grain:
            if os.path.isfile(key.replace('_k', '_1')):
                if self.middle_grain:
                    if key not in self.middle_grain or len(self.middle_grain[key]) == 0:
                        print('Cannot find middle grained data of ' + key)
                        continue
                if self.fine_grain:
                    if key not in self.fine_grain or len(self.fine_grain[key]) == 0:
                        print('Cannot find fine grained data of ' + key)
                        continue
                self.data_idx.append(key)
            else:
                print('Cannot find image ' + key)
                continue
        self.transform = transform
        self.imsize = imsize

        self.berttokenizer = BertTokenizer.from_pretrained("nghuyong/ernie-health-zh")

        self.gpttokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")

        self.text0_max_words = text0_max_words
        self.text1_max_words = text1_max_words
        self.text2_max_words = text2_max_words

    def __len__(self):
        return len(self.data_idx)

    def get_caption(self, sent):
        tokens = self.berttokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.text0_max_words,
        )

        return tokens['input_ids'][0], tokens['attention_mask'][0]

    def get_instruction(self, sent, prompt_length, split):
        if split == 'coarse':
            tokens = self.gpttokenizer(
                sent,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=self.text0_max_words,
            )
        elif split == 'middle':
            tokens = self.gpttokenizer(
                sent,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=self.text1_max_words,
            )
        else:
            tokens = self.gpttokenizer(
                sent,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=self.text2_max_words,
            )

        label = tokens['input_ids'][0].clone()
        label[:prompt_length] = -100
        sent_len = int(torch.where(label == self.gpttokenizer.sep_token_id)[0])
        label[sent_len+1:] = -100
        return tokens['input_ids'][0], tokens['attention_mask'][0], label

    def get_img(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.imsize, self.imsize))
        img = self.transform(img)
        return img

    def __getitem__(self, idx):
        filename = self.data_idx[idx]
        img0_path = filename.replace('_k', '_1')
        img1_path = filename.replace('_k', '_2')
        img0 = self.get_img(img0_path)
        if os.path.isfile(img1_path):
            img1 = self.get_img(img1_path)
        else:
            img1 = img0

        report, attn = self.get_caption(self.coarse_grain[filename][0][1])

        text0, attn0, label0 = self.get_instruction(self.coarse_grain[filename][0][0] + self.coarse_grain[filename][0][1], len(self.coarse_grain[filename][0][0])+1,split='coarse')

        if self.middle_grain:
            prompt, response = random.choice(self.middle_grain[filename])
            text1, attn1, label1 = self.get_instruction(prompt + response,
                                                 len(prompt)+1, split='middle')
        else:
            text1, attn1, label1 = None, None, None

        if self.fine_grain:
                prompt, response = random.choice(self.fine_grain[filename])
                text2, attn2, label2 = self.get_instruction(prompt + response,
                                                 len(prompt)+1, split='fine')
        else:
            text2, attn2, label2 = None, None, None

        return img0, img1, report, attn, text0, attn0, label0, text1, attn1, label1, text2, attn2, label2

def multimodal_collate_fn(batch):
    """sort sequence"""
    img0s, img1s, reports, attns, text0s, attn0s, label0s, text1s, attn1s, label1s, text2s, attn2s, label2s = [], [], [], [], [], [], \
        [], [], [], [], [], [], []
    for b in batch:
        img0, img1, report, attn, text0, attn0, label0, text1, attn1, label1, text2, attn2, label2 = b
        img0s.append(img0)
        img1s.append(img1)
        reports.append(report)
        attns.append(attn)
        text0s.append(text0)
        attn0s.append(attn0)
        label0s.append(label0)
        if text1 != None:
            text1s.append(text1)
            attn1s.append(attn1)
            label1s.append(label1)
        else:
            text1s = None
            attn1s = None
            label1s = None
        if text2 != None:
            text2s.append(text2)
            attn2s.append(attn2)
            label2s.append(label2)
        else:
            text2s = None
            attn2s = None
            label2s = None

    # stack
    img0s = torch.stack(img0s)
    img1s = torch.stack(img1s)
    reports = torch.stack(reports)
    attns = torch.stack(attns)
    text0s = torch.stack(text0s)
    attn0s = torch.stack(attn0s)
    label0s = torch.stack(label0s)
    if text1s:
        text1s = torch.stack(text1s)
        attn1s = torch.stack(attn1s)
        label1s = torch.stack(label1s)
    if text2s:
        text2s = torch.stack(text2s)
        attn2s = torch.stack(attn2s)
        label2s = torch.stack(label2s)

    return_dict = {
        'img0': img0s,
        'img1': img1s,
        'report': reports,
        'attn': attns,
        'text0': text0s,
        'attn0': attn0s,
        'label0': label0s,
        'text1': text1s,
        'attn1': attn1s,
        'label1': label1s,
        'text2': text2s,
        'attn2': attn2s,
        'label2': label2s
    }
    return return_dict


if __name__ == "__main__":
    from datasets.transforms import DataTransforms
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    transform = DataTransforms(is_train=True)
    dataset = MultimodalPretrainingDataset(split="val", transform=transform)
    dataloader = DataLoader(dataset, batch_size=3, collate_fn=multimodal_collate_fn)
    for i in enumerate(tqdm(dataloader)):
        pass
