"""
Created on Mar 04, 2018

@author: Siyuan Qi

Description of the file.

"""

import os
import shutil
import cv2

import numpy as np
import scipy.misc
import vsrl_utils as vu
import matplotlib.pyplot as plt
import json
import pandas as pd

from collections import OrderedDict

plt.switch_backend('agg')

vcoco_label_map = ['hold', 'sit', 'ride', 'look', 'hit', 'eat', 'jump', 'lay', 'talk_on_phone', 'carry', 'throw',
                   'catch', 'cut', \
                   'work_on_computer', 'ski', 'surf', 'skateboard', 'drink', 'kick', 'point', 'read', 'snowboard']
def GetFeatureMapRoi(img_size, img_bbox, feature_map_size):
    '''
    img_size : (w, h, d)
    img_bbox : (x1, x2, y1, y2)
    feature_map_size : (w, h)
    '''
    img_w, img_h = img_size[0], img_size[1]
    fm_w, fm_h = feature_map_size[0], feature_map_size[1]
    roi_x1 = (img_bbox[0] / img_w) * fm_w
    roi_x2 = (img_bbox[2] / img_w) * fm_w
    roi_y1 = (img_bbox[1] / img_h) * fm_h
    roi_y2 = (img_bbox[3] / img_h) * fm_h

    roi = np.round_((roi_x1, roi_x2, roi_y1, roi_y2), decimals=2)

    return roi.tolist()

def plot_box_with_label(img, box, color, label):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, label, tuple(box[:2].tolist()), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.rectangle(img, tuple(box[:2].tolist()), tuple(box[2:].tolist()), color)
    return img


def plot_set(paths, imageset):
    imageset = imageset
    hbbox = OrderedDict()
    obbox = OrderedDict()
    olabel = OrderedDict()
    actions = OrderedDict()
    filenames = []
    imgsizes = OrderedDict()

    vcoco_imageset = 'val' if imageset == 'test' else 'train'

    #
    image_folder = os.path.join('/home/priv-lab1/Database/MSCOCO2014/' '{}2014'.format(vcoco_imageset))
    result_folder = os.path.join('./tmp', 'results/VCOCO/detections/gt')

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    coco = vu.load_coco()

    vcoco_all = vu.load_vcoco('vcoco_{}'.format(imageset))

    # for k,v in vcoco_all[0].items():
    #     print(k,v)

    # agents=[]
    # roles=[]
    # for i,data in enumerate(vcoco_all):
    #     agents.append(datla['action_name'])
    #     roles.append(data['role_name'])
    # with open('action.txt','w') as f:
    #     for i,agent in enumerate(agents):
    #         f.write(agent+' '+str(roles[i])+'\n')

    image_ids = vcoco_all[0]['image_id']
    image_info_list = coco.loadImgs(ids=image_ids[:, 0].tolist())  # 保存的是每张图的信息，是个字典。
    image_ann_count = dict()
    #
    for i_action, vcoco in enumerate(vcoco_all):
        print("the {}th action".format(i_action))

        vcoco = vu.attach_gt_boxes(vcoco, coco)
        positive_indices = np.where(vcoco['label'] == 1)[0].tolist()

        for image_i in positive_indices:  # 表示有关系的图像
            # print(vcoco['ann_id'][image_i])

            human_ann = vcoco['ann_id'][image_i][0]
            anns = coco.loadAnns(ids=[human_ann])[0]

            # img_id = vcoco['image_id'][image_i, 0]
            img_name = image_info_list[image_i]['file_name']  #
            img_size = [image_info_list[image_i]['width'], image_info_list[image_i]['height'], 3]

            # 第几张图其实就是第几个图的id
            image_path = os.path.join(image_folder, img_name)

            assert os.path.exists(image_path)
            # 以下两行代码用来提取框，注意是提取每类hoi对应的人、物的框
            role_bbox = vcoco['role_bbox'][image_i, :] * 1.
            role_bbox = role_bbox.reshape((-1, 4))

            if len(role_bbox) == 1:  # 对于agent类，不统计其hoi对
                continue

            elif len(role_bbox) == 2:
                action_name = vcoco_label_map.index(vcoco['action_name'])  # action_name
                if not np.isnan(role_bbox[1, 0]) and vcoco['role_object_id'][image_i][1] != 0:
                    # 先提取的人的框
                    if img_name not in hbbox:
                        hbbox[img_name] = [role_bbox[0, :].astype(int).tolist()]
                    else:
                        hbbox[img_name].append(role_bbox[0, :].astype(int).tolist())
                    # 再根据role_object来提取物体的框
                    if img_name not in obbox:
                        obbox[img_name] = [role_bbox[1, :].astype(int).tolist()]
                    else:
                        obbox[img_name].append(role_bbox[1, :].astype(int).tolist())

                    label = coco.loadAnns(ids=[vcoco['role_object_id'][image_i][1]])[0]['category_id']
                    if not img_name in olabel:
                        olabel[img_name] = [label]
                    else:
                        # 主要的是针对不同的人体实例对应hoi，所以对应一张图片可能对应多个的hoi对
                        olabel[img_name].append(label)
                    if img_name not in actions:
                        actions[img_name] = [action_name]
                    else:
                        actions[img_name].append(action_name)

                    if img_name not in imgsizes:
                        imgsizes[img_name] = img_size
                    if img_name not in filenames:
                        filenames.append(img_name)


            elif len(role_bbox) == 3:
                action_name = vcoco_label_map.index(vcoco['action_name'])  # action_name
                # 先提取的人的框,注意在此类的hoi中，我们不头
                if (not np.isnan(role_bbox[1, 0]) and not np.isnan(role_bbox[2, 0])) and \
                        (vcoco['role_object_id'][image_i][1] != 0 and vcoco['role_object_id'][image_i][2] != 0):
                    if img_name not in hbbox:
                        hbbox[img_name] = [role_bbox[0, :].astype(int).tolist()]
                    else:
                        hbbox[img_name].append(role_bbox[0, :].astype(int).tolist())
                    if img_name not in obbox:
                        obbox[img_name] = [role_bbox[1, :].astype(int).tolist()]
                    else:
                        obbox[img_name].append(role_bbox[1, :].astype(int).tolist())
                    label_ann = coco.loadAnns(ids=[vcoco['role_object_id'][image_i][1]])  # [0]['category_id']
                    label = label_ann[0]['category_id']
                    if not img_name in olabel:
                        olabel[img_name] = [label]
                    else:
                        olabel[img_name].append(label)
                    if img_name not in actions:
                        actions[img_name] = [action_name]
                    else:
                        actions[img_name].append(action_name)
                    if img_name not in imgsizes:
                        imgsizes[img_name] = img_size
                    if img_name not in filenames:
                        filenames.append(img_name)

    json.dump(hbbox, open('./vcoco_hbbox_new.json', 'w'), indent=4)
    json.dump(obbox, open('./vcoco_obbox_new.json', 'w'), indent=4)
    json.dump(olabel, open('./vcoco_olabel_new.json', 'w'), indent=4)
    json.dump(actions, open('./vcoco_actions_new.json', 'w'), indent=4)
    json.dump(imgsizes, open('./vcoco_imgsizes_new.json', 'w'), indent=4)

    with open('./train_filename.txt', 'w') as f:
        for file in filenames:
            f.write(file + '\n')


def make_multi_json(mode):
    with open('./vcoco_obbox_new.json'.format(mode), 'r') as f:
        object_bbox_dict = json.load(f)
    with open('./vcoco_hbbox_new.json'.format(mode), 'r') as f:
        human_bbox_dict = json.load(f)
    with open('./vcoco_actions_new.json'.format(mode), 'r') as f:
        action_dict = json.load(f)
    with open("./{}_filename.txt".format(mode, mode), 'r') as f:
        lines = f.readlines()
    filenames = []
    for line in lines:
        filenames.append(line.strip())

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
    num = 0

    for i, file in enumerate(filenames):
        print(file)
        actions = action_dict[file]
        obboxs = object_bbox_dict[file]
        hbboxs = human_bbox_dict[file]

        obboxs_arr = np.array(object_bbox_dict[file])
        hbboxs_arr = np.array(human_bbox_dict[file])
        # hbboxs_arr[:, [1, 2]] = hbboxs_arr[:, [2, 1]]
        # obboxs_arr[:, [1, 2]] = obboxs_arr[:, [2, 1]]
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
        num = num + len(hbbox_sim_dict)
        print(i, action_sim_dict)
        print(i, hbbox_sim_dict)

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

def make_ori_csv(mode):
    with open('./vcoco_obbox_new.json'.format(mode), 'r') as f:
        object_bbox_dict = json.load(f)
    with open('./vcoco_hbbox_new.json'.format(mode), 'r') as f:
        human_bbox_dict = json.load(f)
    with open('./vcoco_actions_new.json'.format(mode), 'r') as f:
        action_dict = json.load(f)
    with open('./vcoco_imgsizes_new.json', 'r') as f:
        img_size_dict = json.load(f)
    with open("{}_filename.txt".format(mode), 'r') as f:
        lines = f.readlines()
    filenames = []
    for line in lines:
        filenames.append(line.strip())
    img_names = []
    img_hoi_act = []
    img_hoi_human_bbox = []
    img_hoi_obj_bbox = []
    img_sizes = []
    num = 0
    print(len(action_dict))
    for file in filenames:
        hbboxs = human_bbox_dict[file]
        actions = action_dict[file]
        obboxs = object_bbox_dict[file]
        num = num + len(actions)
        for i, action in enumerate(actions):
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

def make_multi_csv(mode):
    with open('./obbox_multi_dict.json'.format(mode), 'r') as f:
        object_bbox_dict = json.load(f)
    with open('./hbbox_multi_dict.json', 'r') as f:
        human_bbox_dict = json.load(f)
    with open('./action_multi_dict.json', 'r') as f:
        action_dict = json.load(f)
    with open('./vcoco_imgsizes_new.json', 'r') as f:
        img_size_dict = json.load(f)
    with open("{}_filename.txt".format(mode), 'r') as f:
        lines = f.readlines()
    filenames = []
    for line in lines:
        filenames.append(line.strip())
    img_names = []
    img_hoi_act = []
    img_hoi_human_bbox = []
    img_hoi_obj_bbox = []
    img_sizes = []
    num = 0
    print(len(action_dict))
    for file in filenames:
        hbboxs = human_bbox_dict[file]
        actions = action_dict[file]
        obboxs = object_bbox_dict[file]
        num = num + len(actions)
        for i, action in enumerate(actions):
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

    df_gt_data['h_44fm_roi_y1x1y2x2'] = df_gt_data['h_44_fm_roi'].apply(lambda x: [x[1], x[0], x[3], x[2]])
    df_gt_data['o_44fm_roi_y1x1y2x2'] = df_gt_data['o_44_fm_roi'].apply(
        lambda x: [x[1], x[0], x[3], x[2]])  # 调换为y1x1y2x2
    df_gt_data.to_csv('./final_multi_{}_data.csv'.format(mode))
    print("{}_final_csv done".format(mode))




def main():
    # paths = vcoco_config.Paths()
    paths = '/home/priv-lab1/workspace/zxh/My_Database/v-coco'
    imagesets = ['train']
    for imageset in imagesets:
        plot_set(paths, imageset)


if __name__ == '__main__':
    main()
    # make_multi_json('train')
    # make_multi_csv('train')
    # make_final_data('train')
    # count_hoi_num('train')
