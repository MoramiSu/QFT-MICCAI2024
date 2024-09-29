import os
from pathlib import Path


DATA_BASE_DIR = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "./data")
DATA_BASE_DIR = Path(DATA_BASE_DIR)
# #############################################


ULTRA_DATA_DIR = DATA_BASE_DIR / 'ultrasound'
ULTRA_MASTER_CSV = ULTRA_DATA_DIR / 'master.csv'
ULTRA_BREAST_RAW = ULTRA_DATA_DIR / 'new_Mammary2.json'
ULTRA_THYROID_RAW = ULTRA_DATA_DIR / 'new_Thyroid2.json'
ULTRA_COARSE_TRAIN = ULTRA_DATA_DIR / 'coarse_train.json'
ULTRA_COARSE_VAL = ULTRA_DATA_DIR / 'coarse_val.json'
ULTRA_COARSE_TEST = ULTRA_DATA_DIR / 'coarse_test.json'
ULTRA_MIDDLE_MASTER = ULTRA_DATA_DIR / 'ultrasound_middle_grained.csv'
ULTRA_MIDDLE_TRAIN = ULTRA_DATA_DIR / 'middle_train.json'
ULTRA_MIDDLE_VAL = ULTRA_DATA_DIR / 'middle_val.json'
ULTRA_MIDDLE_TEST = ULTRA_DATA_DIR / 'middle_test.json'
ULTRA_FINE_MASTER = ULTRA_DATA_DIR / 'ultrasound_fine_grained.csv'
ULTRA_FINE_TRAIN = ULTRA_DATA_DIR / 'fine_train.json'
ULTRA_FINE_VAL = ULTRA_DATA_DIR / 'fine_val.json'
ULTRA_FINE_TEST = ULTRA_DATA_DIR / 'fine_test.json'
ULTRA_FINE = ULTRA_DATA_DIR / 'fine.json'
ULTRA_IMG_DIR = ULTRA_DATA_DIR / 'files'
ULTRA_CAPTION_CSV = ULTRA_DATA_DIR / 'caption_master.csv'
ULTRA_PATH_COL = 'Path'
ULTRA_SPLIT_COL = 'split'
ULTRA_GEN_DIR = DATA_BASE_DIR / '../models/textgen'

BUSI_DATA_DIR = DATA_BASE_DIR / 'Dataset_BUSI_with_GT'
BUSI_IMG_DIR = BUSI_DATA_DIR / 'files'
BUSI_CLASSIFICATION_TRAIN_CSV = BUSI_DATA_DIR / 'classification/train.csv'
BUSI_CLASSIFICATION_VALID_CSV = BUSI_DATA_DIR / 'classification/val.csv'
BUSI_CLASSIFICATION_TEST_CSV = BUSI_DATA_DIR / 'classification/test.csv'
BUSI_DETECTION_TRAIN_PKL = BUSI_DATA_DIR / 'detection/train.pkl'
BUSI_DETECTION_TEST_PKL = BUSI_DATA_DIR / 'detection/test.pkl'
BUSI_DETECTION_VALID_PKL = BUSI_DATA_DIR / 'detection/val.pkl'
BUSI_SEG_TRAIN_PKL = BUSI_DATA_DIR / 'detection/train.pkl'
BUSI_SEG_TEST_PKL = BUSI_DATA_DIR / 'detection/test.pkl'
BUSI_SEG_VALID_PKL = BUSI_DATA_DIR / 'detection/val.pkl'
BUSI_ROC = DATA_BASE_DIR / '../data/busi_rocdata.csv'
BUSI_DETECTION_VISUAL_RES_QFT = DATA_BASE_DIR / '../data/Busi_det_res_qft'
BUSI_SEG_VISUAL_RES_QFT = DATA_BASE_DIR / '../data/Busi_seg_res_qft'
BUSI_CLS_VISUAL_RES_QFT = DATA_BASE_DIR / '../data/Busi_cls_res_qft'
BUSI_SEG_VISUAL_RES_GLORIA = DATA_BASE_DIR / '../data/Busi_seg_res_gloria'
BUSI_DETECTION_VISUAL_RES_GLORIA = DATA_BASE_DIR / '../data/Busi_det_res_gloria'
BUSI_CLS_VISUAL_RES_GLORIA = DATA_BASE_DIR / '../data/Busi_cls_res_gloria'


AUIDT_DATA_DIR = DATA_BASE_DIR / 'dataset thyroid'
AUIDT_IMG_DIR = AUIDT_DATA_DIR / 'files'
AUIDT_TRAIN_CSV = AUIDT_DATA_DIR / 'train.csv'
AUIDT_VALID_CSV = AUIDT_DATA_DIR / 'val.csv'
AUIDT_TEST_CSV = AUIDT_DATA_DIR / 'test.csv'

DDTI_DATA_DIR = DATA_BASE_DIR / 'DDTI'
DDTI_IMG_DIR = DDTI_DATA_DIR / 'files'
DDTI_DETECTION_TRAIN_PKL = DDTI_DATA_DIR / 'detection/train.pkl'
DDTI_DETECTION_TEST_PKL = DDTI_DATA_DIR / 'detection/test.pkl'
DDTI_DETECTION_VALID_PKL = DDTI_DATA_DIR / 'detection/val.pkl'
DDTI_SEG_TRAIN_PKL = DDTI_DATA_DIR / 'detection/train.pkl'
DDTI_SEG_TEST_PKL = DDTI_DATA_DIR / 'detection/test.pkl'
DDTI_SEG_VALID_PKL = DDTI_DATA_DIR / 'detection/val.pkl'





