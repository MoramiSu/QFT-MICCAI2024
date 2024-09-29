import json
import pandas as pd
from constants import *

new_breast_data = '../data/ultrasound/new_Mammary2.json'
new_thyroid_data = '../data/ultrasound/new_Thyroid2.json'
coarse_data = '../data/ultrasound/coarse.json'

# def coarse_preprocess():
#     data = pd.read_csv(ULTRA_MASTER_CSV)
#
#     corase_data = {}
#     for _, row in data.iterrows():
#         if '乳腺' in row['findings'] or '甲状腺' in row['findings']:
#             filepath = row['Path'].replace('Ultrasonic_datasets/files', str(ULTRA_IMG_DIR)).replace('jpg', 'jpeg')
#             corase_data[filepath] = [['生成中文超声报告：', row['findings']]]
#
#     with open(ULTRA_COARSE, 'w', encoding='utf-8') as f:
#         json.dump(corase_data, f, ensure_ascii=False)

def partition():
    coarse_train = {}
    with open(new_breast_data, 'r', encoding='utf-8-sig') as f:
        breast_data = json.load(f)
    with open(new_thyroid_data, 'r', encoding='utf-8-sig') as f:
        thyroid_data = json.load(f)
    with open(coarse_data, 'r', encoding='utf-8') as f:
        coarse = json.load(f)

    # for item in breast_data['train']:
    #     filepath = '/home/sutongkun/VLPv2/MGCA/data/ultrasound/files/' + str(item['uid']) + '_k.jpeg'
    #     if filepath in coarse:
    #         coarse_train[filepath] = coarse[filepath]
    #     else:
    #         print('Can not find' + filepath)
    data = {}
    path = []
    report = []
    for item in breast_data['test']:
        filepath = '/home/sutongkun/VLPv2/MGCA/data/ultrasound/files/' + str(item['uid']) + '_k.jpeg'
        if filepath in coarse:
            # coarse_train[filepath] = coarse[filepath]
            pass
        else:
            path.append(filepath)
            report.append(item['finding'].replace('_2DS_', 'x').replace('_3DS_', 'x').replace('_Loc_', 'x').replace('_SCM_', 'x').replace('_SMM_', 'x').replace('_LocR_', 'x'))
    data['path'] = path
    data['finding'] = report
    df = pd.DataFrame(data)
    df.to_csv('./1.csv', encoding='utf-8-sig')

    # with open(ULTRA_COARSE_TRAIN, 'w', encoding='utf-8') as f:
    #     json.dump(coarse_train, f, ensure_ascii=False)

def coarse_preprocess_new():
    with open(new_breast_data, 'r', encoding='utf-8-sig') as f:
        breast_data = json.load(f)
    with open(new_thyroid_data, 'r', encoding='utf-8-sig') as f:
        thyroid_data = json.load(f)
    coarse_data = {}

    for item in breast_data['train']:
        filepath = '/home/sutongkun/VLPv2/QFT/data/ultrasound/files/' + str(item['uid']) + '_k.jpeg'
        report = item['finding'].replace('_2DS_', 'x').replace('_3DS_', 'x').replace('_Loc_', 'x').replace('_SCM_', 'x').replace('_SMM_', 'x').replace('_LocR_', 'x')
        coarse_data[filepath] = [['生成中文超声报告：', report]]

    for item in thyroid_data['train']:
        filepath = '/home/sutongkun/VLPv2/QFT/data/ultrasound/files/' + str(item['uid']) + '_k.jpeg'
        report = item['finding'].replace('_2DS_', 'x').replace('_3DS_', 'x').replace('_Loc_', 'x').replace('_SCM_', 'x').replace('_SMM_', 'x').replace('_LocR_', 'x')
        coarse_data[filepath] = [['生成中文超声报告：', report]]

    with open(ULTRA_COARSE_TRAIN, 'w', encoding='utf-8') as f:
        json.dump(coarse_data, f, ensure_ascii=False)

def middle_preprocess():
    data = pd.read_csv(ULTRA_MIDDLE_MASTER)
    with open(new_breast_data, 'r', encoding='utf-8-sig') as f:
        breast_data = json.load(f)
    with open(new_thyroid_data, 'r', encoding='utf-8-sig') as f:
        thyroid_data = json.load(f)

    i = 1
    middle_data = {}
    for _, row in data.iterrows():
        if '乳腺' in row['findings'] or '甲状腺' in row['findings']:
            filepath = row['Path'].replace('Ultrasonic_datasets/files', str(ULTRA_IMG_DIR)).replace('jpg', 'jpeg')
            middle_data[filepath] = []
            for i in range(1, 5):
                prompt = 'prompt' + str(i)
                label = 'label' + str(i)
                if row[prompt] != ' ' and row[label] != ' ':
                    if row[label][-1] != '。':
                        row[label] += '。'
                    middle_data[filepath].append([row[prompt]+'：', row[label]])

    middle_train = {}
    for item in breast_data['train']:
        filepath = '/home/sutongkun/VLPv2/QFT/data/ultrasound/files/' + str(item['uid']) + '_k.jpeg'
        if filepath in middle_data:
            middle_train[filepath] = middle_data[filepath]
        else:
            pass

    for item in thyroid_data['train']:
        filepath = '/home/sutongkun/VLPv2/QFT/data/ultrasound/files/' + str(item['uid']) + '_k.jpeg'
        if filepath in middle_data:
            middle_train[filepath] = middle_data[filepath]

    with open(ULTRA_MIDDLE_TRAIN, 'w', encoding='utf-8') as f:
        json.dump(middle_train, f, ensure_ascii=False)

def fine_preprocess():
    data = pd.read_csv(ULTRA_FINE_MASTER)
    with open(new_breast_data, 'r', encoding='utf-8-sig') as f:
        breast_data = json.load(f)
    with open(new_thyroid_data, 'r', encoding='utf-8-sig') as f:
        thyroid_data = json.load(f)

    fine_data = {}
    for _, row in data.iterrows():
        if '乳腺' in row['findings'] or '甲状腺' in row['findings']:
            filepath = row['Path'].replace('Ultrasonic_datasets/files', str(ULTRA_IMG_DIR)).replace('jpg', 'jpeg')
            fine_data[filepath] = []
            for i in range(1, 6):
                prompt = 'prompt' + str(i)
                label = 'label' + str(i)
                if row[prompt] != ' ' and row[label] != ' ':
                    # print(filepath)
                    if row[label][-1] != '。':
                        row[label] += '。'
                    fine_data[filepath].append([row[prompt], row[label]])

    fine_train = {}
    for item in breast_data['train']:
        filepath = '/home/sutongkun/VLPv2/QFT/data/ultrasound/files/' + str(item['uid']) + '_k.jpeg'
        if filepath in fine_data:
            fine_train[filepath] = fine_data[filepath]

    for item in thyroid_data['train']:
        filepath = '/home/sutongkun/VLPv2/QFT/data/ultrasound/files/' + str(item['uid']) + '_k.jpeg'
        if filepath in fine_data:
            fine_train[filepath] = fine_data[filepath]

    with open(ULTRA_FINE_TRAIN, 'w', encoding='utf-8') as f:
        json.dump(fine_train, f, ensure_ascii=False)

def create_master():
    with open(new_breast_data, 'r', encoding='utf-8-sig') as f:
        breast_data = json.load(f)
    with open(new_thyroid_data, 'r', encoding='utf-8-sig') as f:
        thyroid_data = json.load(f)

    dict = {'path' : [],
            'finding' : [],
            'split' : []}
    for s in ['train', 'val']:
        for item in breast_data[s]+thyroid_data[s]:
            p = 'E:/VLPv2/Code/GLoRIA/gloria/DS/ultrasound/files/' + str(item['uid']) + '_1.jpeg'
            dict['path'].append(p)
            dict['finding'].append(item['finding'].replace('_2DS_', 'x').replace('_3DS_', 'x').replace('_Loc_', 'x').replace('_SCM_', 'x').replace('_SMM_', 'x').replace('_LocR_', 'x'))
            dict['split'].append(item['split'])

    df = pd.DataFrame(dict)
    df.to_csv(path_or_buf='/home/sutongkun/VLPv2/GLoRIA/gloria/DS/ultrasound/master.csv', index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    coarse_preprocess_new()
    # middle_preprocess()
    # fine_preprocess()
