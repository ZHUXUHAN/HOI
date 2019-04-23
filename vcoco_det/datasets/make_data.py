import pandas as pd
import os
import numpy as np
from collections import OrderedDict
import json
from PIL import Image
import cv2


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


def make_groundtruth_txt(csv_file, txt_dir, classes_num):
    """
    csv_file : your data's groud_truth csv_file
    txt_dir :where your groud_truth saves
    class_num :to get the lanbel_map
    just make ground_truth txt for every file in test, and
    calculate map
    :return:

    """
    df_gt_data = pd.DataFrame.from_csv(csv_file)
    df_gt_data['human_bbox'] = df_gt_data['human_bbox'].apply(lambda x: list(map(int, x.strip('[]').split(','))))
    df_gt_data['obj_bbox'] = df_gt_data['obj_bbox'].apply(lambda x: list(map(int, x.strip('[]').split(','))))
    df_gt_data['img_size_w_h'] = df_gt_data['img_size_w_h'].apply(lambda x: list(map(int, x.strip('[]').split(','))))
    human_bbox_dict = OrderedDict()
    object_bbox_dict = OrderedDict()
    action_dict = OrderedDict()
    filenames = []
    labels = []
    label_map = {}

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
    print("data load done")

    for file in filenames:
        actions = action_dict[file]
        for i, action in enumerate(actions):
            labels.append(action)
    label_copy = sorted(set(labels))

    if max(label_copy) > classes_num:
        label_map = {
            v: i
            for i, v in enumerate(label_copy)
        }
    print(label_map)
    for file in filenames:
        obboxs = object_bbox_dict[file]
        actions = action_dict[file]
        humans = human_bbox_dict[file]
        txtname = file.split('.')[0] + '.txt'
        if os.path.exists(txt_dir + txtname):
            os.remove(txt_dir + txtname)
        for i, hbbox in enumerate(humans):
            h_x1 = hbbox[0]
            h_x2 = hbbox[2]
            h_y1 = hbbox[1]
            h_y2 = hbbox[3]
            o_x1 = obboxs[i][0]
            o_x2 = obboxs[i][2]
            o_y1 = obboxs[i][1]
            o_y2 = obboxs[i][3]
            label = actions[i]
            if not os.path.exists(txt_dir):
                os.makedirs(txt_dir)


            with open(txt_dir + txtname, 'a') as f:
                f.write(str(label) + ' ' + str(h_x1) + ' ' + str(h_y1) + ' ' + str(h_x2) + ' ' + str(
                    h_y2) + ' ' + str(o_x1) + ' ' + str(o_y1) + ' ' + str(o_x2) + ' ' + str(o_y2) + '\n')
    print("save into txt")


def make_filename_txt(mode):
    """
    make filenames for training/test

    """
    df_gt_data = pd.DataFrame.from_csv('./data/anno_box_{}.csv'.format(mode))
    filenames = []
    for index, row in df_gt_data.iterrows():
        if row['name'] in filenames:
            pass
        else:
            filenames.append(row['name'])
    with open("./data/hico_filename_{}.txt".format(mode), 'w') as f:
        for file in filenames:
            f.write(file + '\n')


def make_anno_box():
    def ToDataframe(mat_data, mode):
        bbox_data = mat_data.copy()

        img_names = []
        img_sizes = []
        img_hoi_human_bbox = []
        img_hoi_obj_bbox = []
        img_hoi_act = []

        fail_img = []

        for data_id in range(len(bbox_data)):
            bbox = bbox_data[data_id]

            img_name = bbox['filename']
            img_size = [int(bbox['size']['width']),
                        int(bbox['size']['height']),
                        int(bbox['size']['depth'])]

            bbox_hois = bbox['hoi']
            if (bbox_hois.ndim == 0):
                bbox_hois = bbox_hois[np.newaxis]

            img_hoi = []
            action_label = []
            # for each HOI type
            for hoi_id in range(len(bbox_hois)):
                bbox_hoi = bbox_hois[hoi_id]

                action = bbox_hoi['id']
                action_label.append(action)
                invis = bbox_hoi['invis']

                # unify format
                if (bbox_hoi['connection'].ndim == 1):
                    bbox_hoi['connection'] = bbox_hoi['connection'][np.newaxis]
                if (bbox_hoi['bboxhuman'].ndim == 0):
                    bbox_hoi['bboxhuman'] = bbox_hoi['bboxhuman'][np.newaxis]
                if (bbox_hoi['bboxobject'].ndim == 0):
                    bbox_hoi['bboxobject'] = bbox_hoi['bboxobject'][np.newaxis]

                # for each connection, get human bbox and object bbox
                for con_id in range(len(bbox_hoi['connection'])):
                    if (len(bbox_hoi['connection'][con_id]) < 2):
                        continue
                    human_bbox_id = bbox_hoi['connection'][con_id][0] - 1
                    obj_bbox_id = bbox_hoi['connection'][con_id][1] - 1

                    human_bbox = bbox_hoi['bboxhuman'][human_bbox_id]
                    obj_bbox = bbox_hoi['bboxobject'][obj_bbox_id]

                    human_bbox = [int(human_bbox['x1']),
                                  int(human_bbox['x2']),
                                  int(human_bbox['y1']),
                                  int(human_bbox['y2'])]

                    obj_bbox = [int(obj_bbox['x1']),
                                int(obj_bbox['x2']),
                                int(obj_bbox['y1']),
                                int(obj_bbox['y2'])]

                    img_names.append(img_name)
                    img_sizes.append(img_size)
                    img_hoi_human_bbox.append(human_bbox)
                    img_hoi_obj_bbox.append(obj_bbox)
                    img_hoi_act.append(action)

        dataFrame = pd.DataFrame({
            'name': img_names,
            'action_no': img_hoi_act,
            'human_bbox': img_hoi_human_bbox,
            'obj_bbox': img_hoi_obj_bbox,
            'img_size_w_h': img_sizes
        })

        return dataFrame

    anno_bbox = sio.loadmat('/home/priv-lab1/workspace/zxh/My_Database/hico_20160224_det' + '/anno_bbox.mat',
                            squeeze_me=True)
    bbox_train = anno_bbox['bbox_train']
    bbox_test = anno_bbox['bbox_test']

    img_hoi_train = ToDataframe(bbox_train, mode='train')
    img_hoi_test = ToDataframe(bbox_test, mode='test')

    img_hoi_train.to_csv('./detection/anno_box_train.csv')
    img_hoi_test.to_csv('./detection/anno_box_test.csv')


def make_final_data(mode):
    """
    make train and test final csv file
    """
    gt_data_path = os.path.join('/home/priv-lab1/workspace/zxh/HOI', "hico_det/data/anno_box_{}.csv".format(mode))

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
    df_gt_data.to_csv('./data/final_{}_data.csv'.format(mode))
    print("{}_final_csv done".format(mode))


def make_det_results_csv():
    """
    make test'csv file from det_results'json for testing
    """
    with open('./results/det_results/det_results.json', 'r') as f:
        csv_dict = json.load(f)
    img_names = []
    img_sizes = []
    img_hoi_human_bboxs = []
    img_hoi_obj_bboxs = []
    img_hoi_hscores = []
    img_hoi_oscores = []
    img_hoi_obj_labels = []
    # write into dict
    names = []
    sizes = []
    human_bboxs = []
    obj_bboxs = []
    hscores = []
    oscores = []
    olabels = []

    for k in sorted(csv_dict.keys()):
        img_hoi_human_bbox = []
        img_hoi_obj_bbox = []
        img_hoi_hscore = []
        img_hoi_oscore = []
        img_hoi_obj_label = []
        print(k)

        for shape in csv_dict[k]:
            if shape['label'] == 'person':
                img_hoi_human_bbox.append(shape['bbox'])
                img_hoi_hscore.append(shape['score'])
            else:
                img_hoi_obj_bbox.append(shape['bbox'])
                img_hoi_oscore.append(shape['score'])
                img_hoi_obj_label.append(shape['label'])

            img_names.append(k)
            img_sizes.append(shape['size'])
            img_hoi_human_bboxs.append(img_hoi_human_bbox)
            img_hoi_hscores.append(img_hoi_hscore)
            img_hoi_obj_bboxs.append(img_hoi_obj_bbox)
            img_hoi_oscores.append(img_hoi_oscore)
            img_hoi_obj_labels.append(img_hoi_obj_label)

    file_num = len(img_names)
    sizes_num = len(img_sizes)
    assert file_num == sizes_num

    for f in range(file_num):
        if (f + 1) % 100 == 0 or f + 1 == file_num:
            print('processing {}/{}'.format(str(f + 1), str(file_num)))
        h_num = len(img_hoi_human_bboxs[f])
        o_num = len(img_hoi_obj_bboxs[f])
        for h in range(h_num):
            for o in range(o_num):
                names.append(img_names[f])
                sizes.append(img_sizes[f])
                human_bboxs.append(img_hoi_human_bboxs[f][h])
                obj_bboxs.append(img_hoi_obj_bboxs[f][o])
                hscores.append(img_hoi_hscores[f][h])
                oscores.append(img_hoi_oscores[f][o])
                olabels.append(img_hoi_obj_labels[f][o])

    dataFrame = pd.DataFrame({
        'name': names,
        'img_size_w_h': sizes,
        'human_bbox': human_bboxs,
        'obj_bbox': obj_bboxs,
        'human_score': hscores,
        'obj_score': oscores,
        'obj_label': olabels
    })

    dataFrame.to_csv('./results/det_results/det_results.csv')

    print("Done")


def crop_people(mode):
    img_hoi_train = pd.DataFrame.from_csv('./data/{}/{}_vcoco.csv'.format(mode,mode))
    data_path = '../DATA/MSCOCO2014/{}2014'.format(mode)
    img_hoi_train['human_bbox'] = img_hoi_train['human_bbox'].apply(
        lambda x: list(map(int, x.strip('[]').split(','))))  # str to tuple(x1,x2,y1,y2)

    for index, row in img_hoi_train[:].iterrows():
        try:
            print("processing", str(index + 1))
            img_path = row['name']
            if not os.path.exists(os.path.join(data_path, img_path)):
                print("the path {} not exits".format(os.path.join(data_path, img_path)))

            im = Image.open(os.path.join(data_path, img_path))
            if len(np.array(im).shape) == 2:
                im3 = im2 = im1 = im.convert('L')
                im = Image.merge('RGB', (im1, im2, im3))

            x1 = row['human_bbox'][0]
            x2 = row['human_bbox'][2]
            y1 = row['human_bbox'][1]
            y2 = row['human_bbox'][3]
            box = (x1, y1, x2, y2)
            region = im.crop(box)
            if not os.path.exists('./data/people_crop_imgs'):
                os.mkdir('./data/people_crop_imgs')
            region.save(os.path.join('./data/people_crop_imgs', '{}.jpg'.format(str(index).zfill(7))))
        except:
            print("error", str(index + 1))
            break


def make_two_class_data(mode):
    """
    make exists interactions and no interactions datasets
    :return:
    """
    hoi_list = []
    hoi_dict = {}
    no_interaction = []
    del_list = []

    with open("./data/origin_lists/hico_list_hoi.txt") as f:
        for line in f:
            hoi_list.append(line)
    hoi_list = hoi_list[2:]
    hoi_list = [item.strip() for item in hoi_list]

    for i in hoi_list:
        hoi_dict[int(i.split()[0])] = [i.split()[2], i.split()[1]]

    for hoi in hoi_dict:
        if hoi_dict[hoi][0] == 'no_interaction':
            no_interaction.append(hoi)
    # print(no_interaction)

    img_hoi = pd.DataFrame.from_csv('./data/hico_data/final_{}_data.csv'.format(mode))
    for i, action_no in enumerate(img_hoi['action_no']):
        if action_no in no_interaction:
            img_hoi['action_no'][i] = 0
            print('line', str(i), img_hoi['action_no'][i])
            del_list.append(i)
        else:
            img_hoi['action_no'][i] = 1  # 如果有标签值的话，就对应的是有交互
            print('line', str(i), img_hoi['action_no'][i])
            # img_hoi_train[i,('action_no')]= [1]
    img_hoi.to_csv('./data/two_classes_{}.csv'.format(mode))


def make_sub_data(mode, action_list, sub_name):
    """
    :param mode:'test' or 'train'
    :param action_list:a list conclude your sub-data number
    :param sub_name: your sub-data'name
    :return:
    """
    img_hoi = pd.DataFrame.from_csv('./data/hico_data/final_{}_data.csv'.format(mode))
    img_hoi = img_hoi[img_hoi.action_no.isin(action_list)]
    img_hoi.reset_index(inplace=True, drop=True)  # 每次修改为csv的dataframe 记得重置索引
    print(img_hoi['name'])
    if os.path.exits('./data/{}'):
        img_hoi.to_csv('./data/sub_{}_data_{}.csv'.format(mode, sub_name))
    print('Done')


def count_hoi_num(mode):
    """
    this is to count every hoi's sample num
    """
    img_hoi = pd.DataFrame.from_csv('./data/final_{}_data.csv'.format(mode))
    hois_dict = {}
    hoi_list = []
    hoi_dict = {}
    with open("./data/origin_lists/hico_list_hoi.txt") as f:
        for line in f:
            hoi_list.append(line)
    hoi_list = hoi_list[2:]
    hoi_list = [item.strip() for item in hoi_list]
    for i in hoi_list:
        hoi_dict[int(i.split()[0])] = [i.split()[2], i.split()[1]]

    for index, row in img_hoi.iterrows():

        if row['action_no'] in hois_dict:

            hois_dict[row['action_no']] = hois_dict[row['action_no']] + 1
        else:
            hois_dict[row['action_no']] = 1
    print(hois_dict)
    hoi_text = ''
    with open('./data/train_hoi_counts.txt', 'a') as f:
        for k, v in hois_dict.items():
            for i in hoi_dict[k]:
                hoi_text = hoi_text + i + ' '
            print(hoi_text)
            f.write(hoi_text + str(v) + '\n')
            hoi_text = ''


def make_80obj_data(mode):
    img_hoi = pd.DataFrame.from_csv(
        '/home/priv-lab1/workspace/zxh/HOI/hico_det/data/hico_data/anno_box_{}.csv'.format(mode))
    img_hoi['obj_bbox'] = img_hoi['obj_bbox'].apply(
        lambda x: list(map(int, x.strip('[]').split(','))))  # str to tuple(x1,x2,y1,y2)
    data_path = '/home/priv-lab1/workspace/zxh/My_Database/hico_20160224_det/images/{}2015'.format(mode)

    with open('./data/hico_data/hico_label.txt') as f:
        llabels = f.readlines()
    labels = []
    obj_dict = {}

    for l in llabels:
        ll = l.strip().split(' ')

        obj = ll[1]
        v = ll[2]
        labels.append([obj, v])

    for index, row in img_hoi.iterrows():
        if index % 100 == 0:
            print("processing_{}_imgs".format(str(index)))
        filename = row['name']
        name = filename.split('.')[0]
        action_id = row['action_no']
        obj = labels[int(action_id) - 1][0]
        verb = labels[int(action_id) - 1][1]
        obbox = row['obj_bbox']
        im = Image.open(os.path.join(data_path, filename))
        if len(np.array(im).shape) == 2:
            im3 = im2 = im1 = im.convert('L')
            im = Image.merge('RGB', (im1, im2, im3))

        x1 = row['obj_bbox'][0]
        x2 = row['obj_bbox'][1]
        y1 = row['obj_bbox'][2]
        y2 = row['obj_bbox'][3]
        box = (x1, y1, x2, y2)
        region = im.crop(box)
        obj_dir = os.path.join('/home/priv-lab1/workspace/zxh/HOI/hico_det/data/obj_crop_imgs', obj)
        if obj in obj_dict:
            obj_dict[obj] = obj_dict[obj] + 1
        else:
            obj_dict[obj] = 0
        if not os.path.exists(obj_dir):
            os.mkdir(obj_dir)
        region.save(os.path.join('/home/priv-lab1/workspace/zxh/HOI/hico_det/data/obj_crop_imgs/{}'.format(obj),
                                 '{}_{}.jpg'.format(name, str(obj_dict[obj]))))


if __name__ == '__main__':
    make_groundtruth_txt('/home/priv-lab1/workspace/zxh/HOI/DATA/v-coco/hoi-data/test/final_test_data.csv','/home/priv-lab1/workspace/zxh/HOI/vcoco_det/results/map/multi_all/ground_truth/',classes_num=26)
    # make_final_data('train')
    # make_filename_txt('train')
    # make_det_results_csv()
    # crop_people('train')
    # make_sub_data('train',[5,33],'5_33')
    # count_hoi_num('train')
    # make_two_class_data('train')
    # make_80obj_data('test')
