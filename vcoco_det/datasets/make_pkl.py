import json
import pickle
import numpy as np
import os
import shutil
import cv2
import scipy.misc
import vsrl_utils as vu
import matplotlib.pyplot as plt
import json
from math import isnan
from collections import OrderedDict
from PIL import Image

OUT_PUTDIR='detections.pkl'
detection = []
coco = vu.load_coco('/home/priv-lab1/workspace/zxh/HOI/DATA/v-coco/data/')
vcoco_all = vu.load_vcoco('/home/priv-lab1/workspace/zxh/HOI/DATA/v-coco/data/vcoco/vcoco_test')
image_ids = vcoco_all[0]['image_id']

image_info_list = coco.loadImgs(ids=image_ids[:, 0].tolist())  # 保存的是每张图的信息，是个字典。
action_classes = ['hold', 'stand', 'sit', 'ride', 'walk', 'look', 'hit', 'eat', 'jump', 'lay', 'talk_on_phone', 'carry',
                  'throw', 'catch', 'cut', 'run', 'work_on_computer', 'ski', 'surf', 'skateboard', 'smile', 'drink',
                  'kick', 'point', 'read', 'snowboard']
action_agent = ['hold_agent', 'smile_agent', 'snowboard_agent', 'surf_agent', 'eat_agent', 'jump_agent', 'catch_agent',
                'ski_agent', \
                'skateboard_agent', 'point_agent', 'stand_agent', 'cut_agent', 'work_on_computer_agent', 'lay_agent',
                'drink_agent', \
                'look_agent', 'read_agent', 'run_agent', 'sit_agent', 'hit_agent', 'walk_agent', 'carry_agent',
                'throw_agent', \
                'kick_agent', 'talk_on_phone_agent', 'ride_agent']
action_role = ['hit_instr', 'catch_obj', 'point_instr', 'cut_instr', 'work_on_computer_instr', 'lay_instr', 'look_obj', \
               'talk_on_phone_instr', 'snowboard_instr', 'smile', 'sit_instr', 'carry_obj', 'throw_obj', 'eat_obj',
               'walk', \
               'skateboard_instr', 'kick_obj', 'cut_obj', 'hold_obj', 'hit_obj', 'drink_instr', 'jump_instr',
               'ride_instr', \
               'stand', 'surf_instr', 'eat_instr', 'run', 'read_obj', 'ski_instr']
action_role_obj=['carry','catch','hold','kick','look','read','throw']
imgs=[]
for i_action, vcoco in enumerate(vcoco_all):
    vcoco = vu.attach_gt_boxes(vcoco, coco)
    positive_indices = np.where(vcoco['label'] == 1)[0].tolist()
    agent_name = vcoco['action_name'] + '_agent'
    print("agent", agent_name,'num',len(positive_indices))

    for image_i in sorted(positive_indices):
        img_name = image_info_list[image_i]['file_name']  #
        if not img_name in imgs:
            imgs.append(img_name)

        dic = {}
        img_name = image_info_list[image_i]['file_name']
        role_bbox = vcoco['role_bbox'][image_i, :] * 1.
        role_bbox = role_bbox.reshape((-1, 4))

        dic['person_box'] = role_bbox[0, :].astype(int).tolist()
        dic['image_id'] = int(os.path.splitext(img_name)[0].split("_")[2])


        # for i in range(29):
        # if action_classes.index(vcoco['action_name'])==i:
        dic[agent_name]=1
        for action in action_agent:
            if action not in dic:
                dic[action] = 0

        # for i in range(29):

        # if action_classes.index(vcoco['action_name']) == i:
        if vcoco['action_name']=='cut' or vcoco['action_name']=='hit':

            # if not np.isnan(role_bbox[1, 0]) :

            obj_out_1=role_bbox[1, :].astype(int).tolist()
            action_instr=vcoco['action_name']+'_instr'
            dic[action_instr] = np.append(obj_out_1, 1)

            # if not np.isnan(role_bbox[2, 0]):

            obj_out_2 = role_bbox[2, :].astype(int).tolist()
            action_obj = vcoco['action_name'] + '_obj'
            dic[action_obj] = np.append(obj_out_2, 1)

        elif vcoco['action_name']=='eat':
            if int(os.path.splitext(img_name)[0].split("_")[2])==293221:
                print(role_bbox)

            obj_out_1=role_bbox[2, :].astype(int).tolist()
            action_instr=vcoco['action_name']+'_instr'
            dic[action_instr] = np.append(obj_out_1, 1)

            obj_out_2 = role_bbox[1, :].astype(int).tolist()
            action_obj = vcoco['action_name'] + '_obj'
            dic[action_obj] = np.append(obj_out_2, 1)

        elif vcoco['action_name']=='run' or vcoco['action_name']=='walk' or vcoco['action_name']=='stand' or vcoco['action_name']=='smile':
            action_instr = vcoco['action_name']
            dic[action_instr] = np.append(np.full(4, np.nan).reshape(1, 4), 1)
        else:
            if int(os.path.splitext(img_name)[0].split("_")[2])==293221:
                print(role_bbox)
            if vcoco['action_name'] in action_role_obj:
                obj_out = role_bbox[1, :].astype(int).tolist()
                action_obj = vcoco['action_name']+'_obj'
                dic[action_obj] = np.append(obj_out, 1)
            else:
                action_str = vcoco['action_name'] + '_instr'
                obj_out = role_bbox[1, :].astype(int).tolist()
                dic[action_str] = np.append(obj_out, 1)
        for action in action_role:
            if action not in dic:
                dic[action] = np.append(np.full(4, np.nan).reshape(1, 4), 0.0)

        detection.append(dic)
print(len(imgs))#4923
print("detection made done")
pickle.dump(detection, open(OUT_PUTDIR, "wb"))











