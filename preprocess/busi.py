import os
import pickle
import re

import pandas as pd
from sklearn.model_selection import train_test_split
from constants import *
import cv2
import numpy as np

def classification():
    filenames = []
    labels = []
    df = pd.DataFrame(columns=['filename', 'label'])
    for file in os.listdir(BUSI_DATA_DIR / 'files'):
        if 'mask' not in file:
            filenames.append(file)
            if 'benign' in file:
                labels.append(1)
            elif 'malignant' in file:
                labels.append(2)
            else:
                labels.append(0)
    df['filename'] = filenames
    df['label'] = labels
    train_df, test_df = train_test_split(df, train_size=0.85, random_state=0)
    train_df, val_df = train_test_split(train_df, train_size=0.8, random_state=0)
    print(train_df.shape, val_df.shape, test_df.shape)

    train_df.to_csv(BUSI_CLASSIFICATION_TRAIN_CSV, index=False)
    val_df.to_csv(BUSI_CLASSIFICATION_VALID_CSV, index=False)
    test_df.to_csv(BUSI_CLASSIFICATION_TEST_CSV, index=False)

def detection():
    filenames = []
    bboxes = []
    df = pd.DataFrame(columns=['filenames', 'bboxes'])
    for file in os.listdir(BUSI_DATA_DIR / 'files'):
        if 'mask' in file:
            im = cv2.imread(str(BUSI_DATA_DIR / ('files'+'/'+file)), cv2.IMREAD_GRAYSCALE)
            filenames.append(re.sub('_mask[_0-9]*', '', file))
            row, col = np.where(im==255)
            if row.size == 0:
                bboxes.append(np.zeros(4))
            else:
                rowt = np.min(row)
                rowb = np.max(row)
                coll = np.min(col)
                colr = np.max(col)
                bboxes.append(np.array([coll, rowt, colr, rowb]))

    df['filenames'] = filenames
    df['bboxes'] = bboxes
    df = df.groupby("filenames", as_index=False).agg(list)
    train_df, test_df = train_test_split(df, train_size=0.85, random_state=0)
    train_df, val_df = train_test_split(train_df, train_size=0.8, random_state=0)
    print(train_df.shape, val_df.shape, test_df.shape)

    saveaspkl(train_df, BUSI_DETECTION_TRAIN_PKL)
    saveaspkl(val_df, BUSI_DETECTION_VALID_PKL)
    saveaspkl(test_df, BUSI_DETECTION_TEST_PKL)

def saveaspkl(df, path):
    filenames = np.array(df['filenames'])
    bboxes = np.array(df['bboxes'])
    with open(path, 'wb') as f:
        pickle.dump([filenames, bboxes], f)


if __name__ == '__main__':
    detection()
    '''
    with open(BUSI_DETECTION_TRAIN_PKL, "rb") as f:
        filenames, bboxes = pickle.load(f)
    for filename in filenames:
        img_path = BUSI_DATA_DIR / ('files' + '/' + filename)
        x = cv2.imread(str(img_path), 0)
        if isinstance(x, type(None)):
            print(filename)
    '''
