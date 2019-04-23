import cv2
from pycocotools.coco import COCO
import numpy as np
import pandas as pd
import os
import math
IMAGE_DIR = '/home/priv-lab1/Database/MSCOCO2014/{}2014'

coco_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', \
                'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                'giraffe', 'backpack', 'umbrella', \
                'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', \
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
                'apple', 'sandwich', 'orange', \
                'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                'dining table', 'toilet', 'tv', 'laptop', \
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
                'book', 'clock', 'vase', 'scissors', \
                'teddy bear', 'hair drier', 'toothbrush']
# 注意官方的coco的类别标注是从1开始的
coco_label_map = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32,
                  33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, \
                  48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76,
                  77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
print(len(coco_classes))


def vis_name_id(csv_file, ids, all_vis=False, mode='train'):
    """
    this is a function to vis  hois according the name id
    in the csvfile,if your ids'long over 1,this can vis all
    hois.

    ids :a list that concluding all your hois'id
    all_vis :if you want to vis all hois in an image,you can
    make the parament to True

    """
    if csv_file:
        img_hoi = pd.DataFrame.from_csv(csv_file)
        img_hoi['human_bbox'] = img_hoi['human_bbox'].apply(
            lambda x: list(map(int, x.strip('[]').split(','))))  # str to tuple(x1,x2,y1,y2)
        img_hoi['obj_bbox'] = img_hoi['obj_bbox'].apply(lambda x: list(map(int, x.strip('[]').split(','))))
    labels = ['hold', 'sit', 'ride', 'look', 'hit', 'eat', 'jump', 'lay', 'talk_on_phone', 'carry', 'throw',
              'catch', 'cut', \
              'work_on_computer', 'ski', 'surf', 'skateboard', 'drink', 'kick', 'point', 'read', 'snowboard']
    action_classes = ['hold', 'stand', 'sit', 'ride', 'walk', 'look', 'hit', 'eat', 'jump', 'lay', 'talk_on_phone',
                      'carry',
                      'throw', 'catch', 'cut', 'run', 'work_on_computer', 'ski', 'surf', 'skateboard', 'smile', 'drink',
                      'kick', 'point', 'read', 'snowboard']

    rgbs = [(0, 0, 255), (0, 255, 0), (255, 90, 0), (0, 255, 20), (255, 0, 255), (30, 40, 250), (60, 40, 250),
            (60, 40, 250)]
    text_x = 0
    text_y = 10
    if all_vis:
        img_path = img_hoi['name'][ids[0]]
        print(img_path)
        assert os.path.exists(os.path.join(IMAGE_DIR.format(mode), img_path))

        img = cv2.imread(os.path.join(IMAGE_DIR.format(mode), img_path))
        name = img_path.split('.')[0]
    for i, id in enumerate(ids):
        if not all_vis:
            img_path = img_hoi['name'][id]
            name = img_path.split('.')[0]
        h_x1 = img_hoi['human_bbox'][id][0]
        h_y1 = img_hoi['human_bbox'][id][1]
        h_x2 = img_hoi['human_bbox'][id][2]
        h_y2 = img_hoi['human_bbox'][id][3]
        action_id = img_hoi['action_no'][id]
        h_x = (h_x1 + h_x2) // 2
        h_y = (h_y1 + h_y2) // 2
        o_x1 = img_hoi['obj_bbox'][id][0]
        o_y1 = img_hoi['obj_bbox'][id][1]
        o_x2 = img_hoi['obj_bbox'][id][2]
        o_y2 = img_hoi['obj_bbox'][id][3]
        o_x = (o_x1 + o_x2) // 2
        o_y = (o_y1 + o_y2) // 2
        try:
            obj = coco_classes[coco_label_map.index(img_hoi['olabel'][id])]
        except:
            obj=' '
        verb = action_classes[int(action_id)]
        text = "{} {}".format(verb, obj)

        if not all_vis:
            img = cv2.imread(os.path.join(IMAGE_DIR.format(mode), img_path))
        cv2.rectangle(img, (h_x1, h_y1), (h_x2, h_y2), rgbs[i], 1)
        cv2.rectangle(img, (o_x1, o_y1), (o_x2, o_y2), rgbs[i], 1)
        if not o_x==0 and o_y==0:
            cv2.line(img, (h_x, h_y), (o_x, o_y), rgbs[i], 1, 1)
        if all_vis:
            text_y += 20
        cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rgbs[i], 1)

        if not all_vis:
            cv2.imwrite('./lib/vis_results/{}.jpg'.format(name + '_' + str(id)), img)
    if all_vis:
        cv2.imwrite('./lib/vis_results/{}.jpg'.format(name), img)


def vis_test_result(txt_file, mode,data_mode):
    with open(txt_file) as f:
        lines = f.readlines()
    labels  = ['hold', 'stand', 'sit', 'ride', 'walk', 'look', 'hit', 'eat', 'jump', 'lay', 'talk_on_phone', 'carry',
                  'throw', 'catch', 'cut', 'run', 'work_on_computer', 'ski', 'surf', 'skateboard', 'smile', 'drink',
                  'kick', 'point', 'read', 'snowboard']

    if mode == 'pre':

        action = []
        text_x = 0
        img_path = os.path.split(txt_file)[1].split('.')[0] + '.jpg'
        name = img_path.split('.')[0]
        name = name + '_' + mode
        img = cv2.imread(os.path.join(IMAGE_DIR.format(data_mode), img_path))
        rgbs = [(0, 0, 255), (255, 0, 30), (255, 90, 0), (0, 255, 20), (255, 0, 255), (255, 0, 255), (255, 0, 255),
                (255, 0, 255), (255, 0, 255), (255, 0, 255), (255, 0, 255)]
        for i, line in enumerate(lines):

            hx1 = int(line.strip().split(' ')[2])
            hy1 = int(line.strip().split(' ')[3])
            hx2 = int(line.strip().split(' ')[4])
            hy2 = int(line.strip().split(' ')[5])
            h_x = (hx1 + hx2) // 2
            h_y = (hy1 + hy2) // 2
            ox1 = int(line.strip().split(' ')[6])
            oy1 = int(line.strip().split(' ')[7])
            ox2 = int(line.strip().split(' ')[8])
            oy2 = int(line.strip().split(' ')[9])
            o_x = (ox1 + ox2) // 2
            o_y = (oy1 + oy2) // 2
            action_id = int(line.strip().split(' ')[0])
            # obj = coco_classes[coco_label_map.index(img_hoi['olabel'][id])]
            verb = labels[int(action_id)]
            text = "{} {}".format(verb, '')

            action.append(text)


            if (i + 1) % 3 == 0:
                img_i = img.copy()
                text_y = 10
                rgb = rgbs[(i + 1) // 5]
                for e,text in enumerate(action):
                    cv2.rectangle(img_i,(hx1,hy1), (hx2, hy2), rgb, 1)
                    cv2.rectangle(img_i, (ox1, oy1), (ox2, oy2), rgb, 1)
                    if not (o_x==0 and o_y==0):
                        cv2.line(img_i, (h_x, h_y), (o_x, o_y), rgb, 1, 1)
                    cv2.putText(img_i, str(text), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rgb, 1)
                    text_y += 20

                text_x += 50
                action = []
                cv2.imwrite('./lib/vis_results/{}.jpg'.format(name + '_' + str(i) ), img_i)

    else:
        action = []
        text_x = 0
        img_path = os.path.split(txt_file)[1].split('.')[0] + '.jpg'
        name = img_path.split('.')[0]
        name = name + '_' + mode
        img = cv2.imread(os.path.join(IMAGE_DIR.format(data_mode), img_path))
        rgbs = [(0, 0, 255), (0, 0, 255), (255, 90, 0), (0, 255, 20), (255, 0, 255), (255, 0, 255),(0, 0, 255),(0,0,250),(0,0,250)]
        for i, line in enumerate(lines):

            hx1 = int(line.strip().split(' ')[1])
            hy1 = int(line.strip().split(' ')[2])
            hx2 = int(line.strip().split(' ')[3])
            hy2 = int(line.strip().split(' ')[4])
            h_x = (hx1 + hx2) // 2
            h_y = (hy1 + hy2) // 2
            ox1 = int(line.strip().split(' ')[5])
            oy1 = int(line.strip().split(' ')[6])
            ox2 = int(line.strip().split(' ')[7])
            oy2 = int(line.strip().split(' ')[8])
            o_x = (ox1 + ox2) // 2
            o_y = (oy1 + oy2) // 2
            action_id = int(line.strip().split(' ')[0])
            # obj = labels[int(action_id)][0]
            verb = labels[int(action_id)]
            text = "{} {}".format(verb, '')
            img_i=img.copy()
            if ox1 == -9223372036854775808:
                continue


            action.append(text)
            if (i + 1) % 1 == 0:
                text_y = 10
                rgb = rgbs[(i + 1) // 1]
                cv2.rectangle(img_i, (hx1, hy1), (hx2, hy2), rgb, 1)
                cv2.rectangle(img_i, (ox1, oy1), (ox2, oy2), rgb, 1)

                if not o_x == 0 and o_y == 0 :
                    cv2.line(img, (h_x, h_y), (o_x, o_y), rgb, 1, 1)

                for text in action:
                    print(text)
                    cv2.putText(img_i, str(text), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, rgb, 1)
                    text_y += 20
                text_x += 60
                action = []
            cv2.imwrite('./lib/vis_results/{}.jpg'.format(name+str(i)), img_i)


if __name__ == '__main__':
    # vis_name_id('/home/priv-lab1/workspace/zxh/HOI/DATA/v-coco/hoi-data/train/train_vcoco.csv', range(0,5), all_vis=True,
    #             mode='train')
    vis_test_result('/home/priv-lab1/workspace/zxh/HOI/vcoco_det/results/map/multi_all/predicted/COCO_val2014_000000000328.txt','pre','val')
    vis_test_result(
        '/home/priv-lab1/workspace/zxh/HOI/vcoco_det/results/map/multi_all/ground_truth/COCO_val2014_000000000328.txt', 'gro' ,'val')
