import json

import torch
import torch.utils.data as data
from constants import *
from transformers import BertTokenizer
import cv2
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class CaptionDataset(data.Dataset):
    def __init__(self, transform=None, prompt='生成中文超声报告：',
                 imsize=256, caption_max_words=200, split='train', data_pct=1.0):
        super().__init__()

        with open(ULTRA_COARSE_TEST, 'r', encoding='utf-8') as f:
            coarse_grain = json.load(f)
        if os.path.exists(ULTRA_MIDDLE_TEST):
            with open(ULTRA_MIDDLE_TEST, 'r', encoding='utf-8') as f:
                middle_grain = json.load(f)
        else:
            middle_grain = None
        if os.path.exists(ULTRA_FINE_TEST):
            with open(ULTRA_FINE_TEST, 'r', encoding='utf-8') as f:
                fine_grain = json.load(f)
        else:
            fine_grain = None

        self.data_idx = []
        for key in coarse_grain:
            if os.path.isfile(key.replace('_k', '_1')):
                if middle_grain:
                    if key not in middle_grain or len(middle_grain[key]) == 0:
                        print('Cannot find middle grained data of ' + key)
                        continue
                if fine_grain:
                    if len(fine_grain[key]) == 0:
                        print('Cannot find fine grained data of ' + key)
                        continue
                self.data_idx.append(key)
            else:
                print('Cannot find image ' + key)
                continue

        data_idx = self.data_idx.copy()
        if prompt:
            self.prompt_data = {}
            for key in data_idx:
                for text in coarse_grain[key]:
                    if text[0] == prompt:
                        self.prompt_data[key] = text[1]
                        break
                if key in self.prompt_data:
                    continue
                if middle_grain:
                    for text in middle_grain[key]:
                        if text[0] == prompt:
                            self.prompt_data[key] = text[1]
                            break
                if key in self.prompt_data:
                    continue
                if fine_grain:
                    for text in fine_grain[key]:
                        if text[0] == prompt:
                            self.prompt_data[key] = text[1]
                            break
                if key in self.prompt_data:
                    continue
                self.data_idx.remove(key)

        self.transform = transform
        self.imsize = imsize

        # self.berttokenizer = BertTokenizer.from_pretrained(
        #     "nghuyong/ernie-health-zh")
        self.berttokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")

        self.gpttokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")

        self.caption_max_words = caption_max_words

    def __len__(self):
        return len(self.data_idx)

    def encode_instruction(self, sent, prompt):
        caption_tokens = self.gpttokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.caption_max_words,
        )
        prompt_tokens = self.gpttokenizer.encode(prompt)[:-1]

        return prompt_tokens, caption_tokens['attention_mask'][0]

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

        caption = self.prompt_data[filename]
        caption = self.gpttokenizer(
            caption,
            return_tensors="pt",  # 返回pytorch tensor
            truncation=True,  # 将过长的句子截断到最大长度
            padding="max_length",  # 将过短的句子填充到最大长度
            max_length=self.caption_max_words,
        )

        return img0, img1, filename, caption['input_ids'][0]

def caption_collate_fn(batch):
    """sort sequence"""
    img0s, img1s, filenames, captions = [], [], [], []
    for b in batch:
        img0, img1, filename, caption = b
        img0s.append(img0)
        img1s.append(img1)
        filenames.append(filename)
        captions.append(caption)

    # stack
    img0s = torch.stack(img0s)
    img1s = torch.stack(img1s)
    captions = torch.stack(captions)

    # sort and add to dictionary
    # sorted_cap_lens, sorted_cap_indices = torch.sort(  # 根据非padding词元数降序排序，得到排序后的词元数和对应索引
    #     torch.tensor(cap_len), 0, True)

    return_dict = {
        'img0': img0s,
        'img1': img1s,
        'filename': filenames,
        'caption': captions
    }
    return return_dict


if __name__ == "__main__":
    from datasets.transforms import DataTransforms
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    transform = DataTransforms(is_train=True)
    dataset = CaptionDataset(transform=transform, prompt='甲状腺左叶：')
    dataloader = DataLoader(dataset, batch_size=3, collate_fn=caption_collate_fn)
    for i in enumerate(tqdm(dataloader)):
        pass
