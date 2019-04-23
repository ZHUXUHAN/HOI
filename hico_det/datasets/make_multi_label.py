import os
import pandas as pd
from collections import OrderedDict
import cv2
import numpy as np
import json


def GetFeatureMapRoi(img_size, img_bbox, feature_map_size):
    '''
    img_size : (w, h, d)
    img_bbox : (x1, x2, y1, y2)
    feature_map_size : (w, h)
    '''
    img_w, img_h = img_size[0], img_size[1]
    fm_w, fm_h = feature_map_size[0], feature_map_size[1]
    roi_x1 = (img_bbox[0] / img_w) * fm_w
    roi_x2 = (img_bbox[1] / img_w) * fm_w
    roi_y1 = (img_bbox[2] / img_h) * fm_h
    roi_y2 = (img_bbox[3] / img_h) * fm_h

    roi = np.round_((roi_x1, roi_x2, roi_y1, roi_y2), decimals=2)

    return roi.tolist()
def make_ori_json(mode):
    df_gt_data = pd.DataFrame.from_csv('/home/priv-lab1/workspace/zxh/HOI/hico_det/data/hico_data/anno_box_{}.csv'.format(mode))
    # str to list
    df_gt_data['human_bbox'] = df_gt_data['human_bbox'].apply(lambda x: list(map(int, x.strip('[]').split(','))))
    df_gt_data['obj_bbox'] = df_gt_data['obj_bbox'].apply(lambda x: list(map(int, x.strip('[]').split(','))))
    df_gt_data['img_size_w_h'] = df_gt_data['img_size_w_h'].apply(lambda x: list(map(int, x.strip('[]').split(','))))

    human_bbox_dict = OrderedDict()
    object_bbox_dict = OrderedDict()
    filenames = []
    action_dict = OrderedDict()



    for index, row in df_gt_data.iterrows():

        if row['name'] in filenames:
            pass
        else:
            filenames.append(row['name'])
        if row['name'] in human_bbox_dict:
            human_bbox_dict[row['name']].append(row['human_bbox'])
            object_bbox_dict[row['name']].append(row['obj_bbox'])
            action_dict[row['name']].append(row['action_no'])
        else:
            human_bbox_dict[row['name']] = [row['human_bbox']]
            object_bbox_dict[row['name']] = [row['obj_bbox']]
            action_dict[row['name']] = [row['action_no']]


    with open('{}_filenames.txt'.format(mode),'w') as f:
        for file in filenames:
            f.write(file+'\n')
    json.dump(action_dict, open('./action_dict.json', 'w'), indent=4)

    json.dump(object_bbox_dict, open('./object_bbox_dict.json', 'w'), indent=4)
    json.dump(human_bbox_dict, open('./human_bbox_dict.json', 'w'), indent=4)


def make_multi_json(mode):
    with open("./data/multi_label/{}/{}_filenames.txt".format(mode,mode), 'r') as f:
        lines = f.readlines()
    filenames = []
    for line in lines:
        filenames.append(line.strip())

    with open('./data/multi_label/{}/object_bbox_dict.json'.format(mode), 'r') as f:
        object_bbox_dict = json.load(f)
    with open('./data/multi_label/{}/human_bbox_dict.json'.format(mode), 'r') as f:
        human_bbox_dict = json.load(f)
    with open('./data/multi_label/{}/action_dict.json'.format(mode), 'r') as f:
        action_dict = json.load(f)

    def bbox_iou(bbox_a, bbox_b):
        """
        以bbox1 = np.array([[50, 50, 100, 100],[80, 80, 125, 125], [90, 90, 125,
        125]]).reshape(3, 4)
             bbox2 = np.array([[60, 60, 120, 120], [40, 40, 60, 60]]).reshape(2, 4)为例
        """
        if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
            raise IndexError

        tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])  # 这里是[3, 1, 2]和[2,2]运算
        # 自动扩充到[3, 2, 2]和[3, 2, 2]运算 下面也一样
        br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])

        area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)  # [3, 2]
        area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)  # [3, ]
        area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)  # [2, ]
        return area_i / (area_a[:, None] + area_b - area_i)

    hbbox_multi_dict = OrderedDict()
    obbox_multi_dict = OrderedDict()
    action_multi_dict = OrderedDict()
    num=0

    for i, file in enumerate(filenames[:36]):
        print(file)
        actions = action_dict[file]
        obboxs = object_bbox_dict[file]
        hbboxs = human_bbox_dict[file]

        obboxs_arr = np.array(object_bbox_dict[file])
        hbboxs_arr = np.array(human_bbox_dict[file])
        hbboxs_arr[:, [1, 2]] = hbboxs_arr[:, [2, 1]]
        obboxs_arr[:, [1, 2]] = obboxs_arr[:, [2, 1]]
        h_iou = bbox_iou(hbboxs_arr, hbboxs_arr)
        o_iou = bbox_iou(obboxs_arr, obboxs_arr)

        action_sim_dict = {}
        hbbox_sim_dict = {}
        obbox_sim_dict = {}
        rows = h_iou.shape[0]
        columns = h_iou.shape[1]
        print(h_iou)
        print(o_iou)
        for row in range(rows):
            action_sim_list = []
            obbox_sim_list = []
            hbbox_sim_list = []

            for column in range(columns):
                if h_iou[row][column] > 0.29 and o_iou[row][column] > 0.4:
                    if actions[column] in action_sim_list:
                        continue

                    action_sim_list.append(actions[column])
                    obbox_sim_list.append(obboxs[row])
                    hbbox_sim_list.append(hbboxs[row])

            action_sim_dict[row] = action_sim_list
            obbox_sim_dict[row] = obbox_sim_list
            hbbox_sim_dict[row] = hbbox_sim_list
        num=num+len(hbbox_sim_dict)
        print(i,action_sim_dict)
        print(i,hbbox_sim_dict)


        actions_list = []
        obbox_list = []
        hbbox_list = []

        for k, v in action_sim_dict.items():

            # if (hbbox_sim_dict[k] in hbbox_list) and (obbox_sim_dict[k] in obbox_list) and v in actions_list:
            #     continue
            # else:
            actions_list.append(v)

            obbox_list.append(obbox_sim_dict[k])
            hbbox_list.append(hbbox_sim_dict[k])
            action_multi_dict[file] = actions_list
            obbox_multi_dict[file] = obbox_list
            hbbox_multi_dict[file] = hbbox_list
    print(num)
    print(len(action_multi_dict))
    json.dump(action_multi_dict, open('./action_multi_dict.json', 'w'), indent=4)
    json.dump(obbox_multi_dict, open('./obbox_multi_dict.json', 'w'), indent=4)
    json.dump(hbbox_multi_dict, open('hbbox_multi_dict.json', 'w'), indent=4)

def make_img_size_json(mode):
    img_hoi = pd.DataFrame.from_csv('/home/priv-lab1/workspace/zxh/HOI/hico_det/data/hico_data/anno_box_{}.csv'.format(mode))
    img_size_dict=OrderedDict()
    for index, row in img_hoi.iterrows():
        img_size_dict[row['name']]=row['img_size_w_h']
    json.dump(img_size_dict, open('./img_size_multi_dict.json', 'w'), indent=4)


def make_multi_csv(mode):
    with open('./obbox_multi_dict.json'.format(mode), 'r') as f:
        object_bbox_dict = json.load(f)
    with open('./hbbox_multi_dict.json', 'r') as f:
        human_bbox_dict = json.load(f)
    with open('./action_multi_dict.json', 'r') as f:
        action_dict = json.load(f)
    with open('./img_size_multi_dict.json', 'r') as f:
        img_size_dict = json.load(f)
    with open("{}_filenames.txt".format(mode), 'r') as f:
        lines = f.readlines()
    filenames = []
    for line in lines:
        filenames.append(line.strip())
    img_names=[]
    img_hoi_act=[]
    img_hoi_human_bbox=[]
    img_hoi_obj_bbox=[]
    img_sizes=[]
    num=0
    print(len(action_dict))
    for file in filenames:
        hbboxs=human_bbox_dict[file]
        actions=action_dict[file]
        obboxs = object_bbox_dict[file]
        num=num+len(actions)



        for i,action in enumerate(actions):
                img_names.append(file)
                img_hoi_act.append(action)
                img_hoi_human_bbox.append(hbboxs[i][0])
                img_hoi_obj_bbox.append(obboxs[i][0])
                img_sizes.append(img_size_dict[file])
    print(len(img_names))
    print(num)

    dataFrame = pd.DataFrame({
        'name': img_names,
        'action_no': img_hoi_act,
        'img_size_w_h': img_sizes,
        'human_bbox': img_hoi_human_bbox,
        'obj_bbox': img_hoi_obj_bbox

    })
    dataFrame.to_csv('{}_multi.csv'.format(mode))
def make_final_data(mode):
    """
    make train and test final csv file
    """
    gt_data_path = os.path.join('./{}_multi.csv'.format(mode))

    df_gt_data = pd.DataFrame.from_csv(gt_data_path)
    print(len(df_gt_data))

    # str to list
    df_gt_data['human_bbox'] = df_gt_data['human_bbox'].apply(lambda x: list(map(int, x.strip('[]').split(','))))
    df_gt_data['obj_bbox'] = df_gt_data['obj_bbox'].apply(lambda x: list(map(int, x.strip('[]').split(','))))
    df_gt_data['img_size_w_h'] = df_gt_data['img_size_w_h'].apply(lambda x: list(map(int, x.strip('[]').split(','))))

    feature_map_size = (7, 7)
    df_gt_data['h_44_fm_roi'] = df_gt_data[['human_bbox', 'img_size_w_h']].apply(
        lambda x: GetFeatureMapRoi(x['img_size_w_h'], x['human_bbox'], feature_map_size),
        axis=1)
    df_gt_data['o_44_fm_roi'] = df_gt_data[['obj_bbox', 'img_size_w_h']].apply(
        lambda x: GetFeatureMapRoi(x['img_size_w_h'], x['obj_bbox'], feature_map_size),
        axis=1)

    df_gt_data['h_44fm_roi_y1x1y2x2'] = df_gt_data['h_44_fm_roi'].apply(lambda x: [x[2], x[0], x[3], x[1]])
    df_gt_data['o_44fm_roi_y1x1y2x2'] = df_gt_data['o_44_fm_roi'].apply(
        lambda x: [x[2], x[0], x[3], x[1]])  # 调换为y1x1y2x2
    df_gt_data.to_csv('./final_multi_{}_data.csv'.format(mode))
    print("{}_final_csv done".format(mode))

if __name__ == '__main__':
    # make_ori_json('test')
    # make_multi_json('train')
    # make_img_size_json('test')
    # make_multi_csv('train')
    make_final_data('train')