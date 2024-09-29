import pandas as pd
import os
from sklearn.model_selection import train_test_split
from constants import *

def rename():
    path1 = AUIDT_DATA_DIR / 'test/benign'
    path2 = AUIDT_IMG_DIR
    filename = os.listdir(path1)
    for file in filename:
        os.rename(os.path.join(path1, file), os.path.join(path2, 'd'+file))

def remove():
    path = AUIDT_DATA_DIR / 'train/benign'
    filename = os.listdir(path)
    for file in filename:
        if 'Copy' in file:
            os.remove(os.path.join(path, file))

def build_csv(split):
    pathb = AUIDT_DATA_DIR / (split+'/benign')
    pathm = AUIDT_DATA_DIR / (split+'/Malignant')
    pathn = AUIDT_DATA_DIR / (split+'/normal thyroid')
    data = pd.DataFrame(columns=['filenames', 'labels'])

    if split == 'train':
        filenames = [('a'+name) for name in os.listdir(pathb)] + [('b'+name) for name in os.listdir(pathm)] + [('c'+name) for name in os.listdir(pathn)]
        labels = [1]*len(os.listdir(pathb))+[2]*len(os.listdir(pathm))+[0]*len(os.listdir(pathn))
        data['filenames'] = filenames
        data['labels'] = labels

        train_df, val_df = train_test_split(data, train_size=0.8, random_state=0)
        print(train_df.shape, val_df.shape)
        train_df.to_csv(AUIDT_TRAIN_CSV, index=False)
        val_df.to_csv(AUIDT_VALID_CSV, index=False)

    if split == 'test':
        filenames = ['d'+name for name in os.listdir(pathb)] + ['e'+name for name in os.listdir(pathm)] + ['f'+name for name in os.listdir(pathn)]
        labels = [1] * len(os.listdir(pathb)) + [2] * len(os.listdir(pathm)) + [0] * len(os.listdir(pathn))
        data['filenames'] = filenames
        data['labels'] = labels

        print(data.shape)
        data.to_csv(AUIDT_TEST_CSV, index=False)

if __name__ == '__main__':
    # rename()
    # remove()
    build_csv('train')
    build_csv('test')

