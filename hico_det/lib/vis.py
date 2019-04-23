import cv2
from pycocotools.coco import COCO
import numpy as np
import pandas as pd
import os
IMAGE_DIR='/home/priv-lab1/workspace/zxh/My_Database/hico_20160224_det/images/{}2015'
DATASETS='5_33'
def vis_json_image():
    annFile = './data/test_hico.json'
    coco = COCO(annFile)
    imgIds = coco.getImgIds()
    print(imgIds)
def vis_name_id(csv_file,ids,all_vis=False,mode='train'):
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

    with open('./data/hico_data/hico_label.txt') as f:
        llabels=f.readlines()
    labels=[]

    for l in llabels:
        ll=l.strip().split(' ')

        obj=ll[1]
        v=ll[2]
        labels.append([obj,v])

    rgbs = [(0, 0, 255), (255, 0, 30), (255, 90, 0), (0, 255, 20), (255, 0, 255),(30,40,250),(60,40,250),(60,40,250)]
    text_x = 0
    text_y = 10
    if all_vis:
        img_path = img_hoi['name'][ids[0]]
        print(img_path)
        img = cv2.imread(os.path.join(IMAGE_DIR.format(mode), img_path))
        name=img_path.split('.')[0]
    for i,id in enumerate(ids):
        if not all_vis:
            img_path = img_hoi['name'][id]
            name = img_path.split('.')[0]
        h_x1 = img_hoi['human_bbox'][id][0]
        h_x2 = img_hoi['human_bbox'][id][1]
        h_y1 = img_hoi['human_bbox'][id][2]
        h_y2 = img_hoi['human_bbox'][id][3]
        action_id=img_hoi['action_no'][id]
        h_x=(h_x1 + h_x2) // 2
        h_y=(h_y1 + h_y2) // 2
        o_x1 = img_hoi['obj_bbox'][id][0]
        o_x2 = img_hoi['obj_bbox'][id][1]
        o_y1 = img_hoi['obj_bbox'][id][2]
        o_y2 = img_hoi['obj_bbox'][id][3]
        o_x=(o_x1 + o_x2) // 2
        o_y=(o_y1 + o_y2) // 2
        obj = labels[int(action_id) - 1][0]
        verb = labels[int(action_id) - 1][1]
        text = "{} {}".format(verb, obj)

        if not all_vis:
            img=cv2.imread(os.path.join(IMAGE_DIR, img_path))
        cv2.rectangle(img,(h_x1,h_y1),(h_x2,h_y2),rgbs[i],1)
        cv2.rectangle(img, (o_x1, o_y1), (o_x2, o_y2), rgbs[i], 1)
        cv2.line(img, (h_x, h_y), (o_x, o_y), rgbs[i], 1, 1)
        if all_vis:
            text_y+=20
        cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rgbs[i], 1)

        if not all_vis:
            cv2.imwrite('./lib/vis_results/{}.jpg'.format(name+'_'+str(id)),img)
    if all_vis:
        cv2.imwrite('./lib/vis_results/{}.jpg'.format(name), img)
def vis_test_result(txt_file,mode):
    with open(txt_file) as f:
        lines=f.readlines()
    with open('/home/priv-lab1/workspace/zxh/HOI/hico_det/data/hico_data/hico_label.txt') as f:
        llabels=f.readlines()
    labels=[]

    for l in llabels:
        ll=l.strip().split(' ')

        obj=ll[1]
        v=ll[2]
        labels.append([obj,v])


    if mode=='pre':

        action=[]
        text_x = 0
        img_path=os.path.split(txt_file)[1].split('.')[0]+'.jpg'
        name = img_path.split('.')[0]
        name=name+'_'+mode
        img = cv2.imread(os.path.join(IMAGE_DIR.format('test'), img_path))
        rgbs = [(0, 0, 255), (255, 0, 30), (255, 90, 0), (0, 255, 20), (255, 0, 255),(255, 0, 255),(255, 0, 255),(255, 0, 255),(255, 0, 255),(255, 0, 255),(255, 0, 255)]
        for i,line in enumerate(lines):
            print(line.strip().split(' '))
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
            action_id=int(line.strip().split(' ')[0])
            obj = labels[int(action_id) ][0]
            verb = labels[int(action_id)][1]
            text = "{} {}".format(verb, obj)

            action.append(text)
            if (i+1)%5==0:
                text_y = 10
                rgb=rgbs[(i+1)//5]
                cv2.rectangle(img, (hx1, hy1), (hx2, hy2), rgb, 1)
                cv2.rectangle(img, (ox1, oy1), (ox2, oy2), rgb, 1)
                cv2.line(img, (h_x, h_y), (o_x, o_y), rgb, 1, 1)
                for text in action:
                    print(text)
                    cv2.putText(img, str(text), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,rgb, 1)
                    text_y += 20
                text_x += 160
                action=[]
        cv2.imwrite('./lib/{}.jpg'.format(name), img)
    else:
        action = []
        text_x = 0
        img_path = os.path.split(txt_file)[1].split('.')[0] + '.jpg'
        name = img_path.split('.')[0]
        name=name + '_'+mode
        img = cv2.imread(os.path.join(IMAGE_DIR.format('test'), img_path))
        rgbs = [(0, 0, 255), (255, 0, 30), (255, 90, 0), (0, 255, 20), (255, 0, 255), (0,255,255),(250,40,50),(0,255,255),(0,255,255),(0,255,255),(0,255,255),(0, 0, 255)]
        for i, line in enumerate(lines):
            print(line.strip().split(' '))
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
            obj = labels[int(action_id) ][0]
            verb = labels[int(action_id) ][1]
            text = "{} {}".format(verb, obj)

            action.append(text)
            if (i + 1) % 1 == 0:
                text_y = 10
                cv2.rectangle(img, (hx1, hy1), (hx2, hy2), rgbs[i], 1)
                cv2.rectangle(img, (ox1, oy1), (ox2, oy2), rgbs[i], 1)
                cv2.line(img, (h_x, h_y), (o_x, o_y), rgbs[i], 1, 1)
                for text in action:
                    print(text)
                    cv2.putText(img, str(text), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, rgbs[i], 1)
                    text_y += 20
                text_x += 140
                action = []
        cv2.imwrite('./lib/{}.jpg'.format(name), img)





if __name__ == '__main__':
    # vis_json_image(
    # vis_name_id('/home/priv-lab1/workspace/zxh/HOI/hico_det/data/hico_data/anno_box_train.csv',range(76,83),all_vis=True,mode='train')
    vis_test_result('/home/priv-lab1/workspace/zxh/HOI/hico_det/results/map/multi_all/predicted/HICO_test2015_00000049.txt','pre')
    vis_test_result(
        '/home/priv-lab1/workspace/zxh/HOI/hico_det/results/map/multi_all/ground_truth/HICO_test2015_00000049.txt', 'gro')

