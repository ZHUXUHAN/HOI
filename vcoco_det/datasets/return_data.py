import pandas as pd
import os
import numpy as np
from PIL import Image
from sklearn.utils import shuffle
import cv2


def return_data(csv_file):
    df_data = pd.DataFrame.from_csv(csv_file)
    df_data['human_bbox'] = df_data['human_bbox'].apply(lambda x: list(map(int, x.strip('[]').split(','))))
    df_data['obj_bbox'] = df_data['obj_bbox'].apply(lambda x: list(map(int, x.strip('[]').split(','))))
    df_data['img_size_w_h'] = df_data['img_size_w_h'].apply(lambda x: list(map(int, x.strip('[]').split(','))))
    df_data['h_44fm_roi_y1x1y2x2'] = df_data['h_44fm_roi_y1x1y2x2'].apply(
        lambda x: list(map(float, x.strip('[]').split(','))))
    df_data['o_44fm_roi_y1x1y2x2'] = df_data['o_44fm_roi_y1x1y2x2'].apply(
        lambda x: list(map(float, x.strip('[]').split(','))))

    # print(df_gt_data.head())
    df_data = df_data[['name',
                       'human_bbox',
                       'obj_bbox',
                       'h_44fm_roi_y1x1y2x2',
                       'o_44fm_roi_y1x1y2x2',
                       'action_no',
                       'olabel']]
    df_data['score'] = 1
    return df_data


def return_multi_data(csv_file):
    df_data = pd.DataFrame.from_csv(csv_file)
    df_data['human_bbox'] = df_data['human_bbox'].apply(lambda x: list(map(int, x.strip('[]').split(','))))
    df_data['obj_bbox'] = df_data['obj_bbox'].apply(lambda x: list(map(int, x.strip('[]').split(','))))
    df_data['img_size_w_h'] = df_data['img_size_w_h'].apply(lambda x: list(map(int, x.strip('[]').split(','))))
    df_data['action_no'] = df_data['action_no'].apply(lambda x: list(map(int, x.strip('[]').split(','))))

    df_data['h_44fm_roi_y1x1y2x2'] = df_data['h_44fm_roi_y1x1y2x2'].apply(
        lambda x: list(map(float, x.strip('[]').split(','))))
    df_data['o_44fm_roi_y1x1y2x2'] = df_data['o_44fm_roi_y1x1y2x2'].apply(
        lambda x: list(map(float, x.strip('[]').split(','))))

    df_data = df_data[['name',
                       'human_bbox',
                       'obj_bbox',
                       'h_44fm_roi_y1x1y2x2',
                       'o_44fm_roi_y1x1y2x2',
                       'action_no']]
    df_data['score'] = 1
    return df_data


def get_map_n_pad(box1, box2, length):  # 第一个box是人的box，第二个box是物体的box
    # get minimum x, y
    x_min = min(box1[0], box2[0])
    y_min = min(box1[1], box2[1])
    # translate
    bx1 = np.array([box1[0] - x_min, box1[1] - y_min,
                    box1[2] - x_min, box1[3] - y_min], dtype=np.float32)
    bx2 = np.array([box2[0] - x_min, box2[1] - y_min,
                    box2[2] - x_min, box2[3] - y_min], dtype=np.float32)
    # get new width and height
    w = max(bx1[2], bx2[2]) - min(bx1[0], bx2[0])
    h = max(bx1[3], bx2[3]) - min(bx1[1], bx2[1])
    # scale
    factor_w = np.float(length) / np.float(w)
    factor_h = np.float(length) / np.float(h)  # 减去最小的x和y
    bx1_rs = np.array([bx1[0] * factor_w, bx1[1] * factor_h,
                       bx1[2] * factor_w, bx1[3] * factor_h])  # 根据default_size rescale human box 到64
    bx2_rs = np.array([bx2[0] * factor_w, bx2[1] * factor_h,
                       bx2[2] * factor_w, bx2[3] * factor_h])

    # generate map
    map_1 = np.zeros([length, length], dtype=np.uint8)
    map_2 = np.zeros([length, length], dtype=np.uint8)
    for i in range(int(round(bx1_rs[1])), int(round(bx1_rs[3]))):
        for j in range(int(round(bx1_rs[0])), int(round(bx1_rs[2]))):
            map_1[i][j] = 1
    for i in range(int(round(bx2_rs[1])), int(round(bx2_rs[3]))):
        for j in range(int(round(bx2_rs[0])), int(round(bx2_rs[2]))):
            map_2[i][j] = 1
    return np.array([map_1, map_2]), bx1_rs, bx2_rs


def get_map_w_pad(box1, box2, length):
    # get minimum x, y
    x_min = min(box1[0], box2[0])
    y_min = min(box1[1], box2[1])

    # translate
    bx1 = np.array([box1[0] - x_min, box1[1] - y_min,
                    box1[2] - x_min, box1[3] - y_min], dtype=np.float32)
    bx2 = np.array([box2[0] - x_min, box2[1] - y_min,
                    box2[2] - x_min, box2[3] - y_min], dtype=np.float32)

    # get new width and height
    w = max(bx1[2], bx2[2]) - min(bx1[0], bx2[0])
    h = max(bx1[3], bx2[3]) - min(bx1[1], bx2[1])
    # scale
    if h > w:
        factor = np.float(length) / np.float(h)
        num_pad = int(round((length - factor * w) / 2.0))
        bx1_rs = np.array([bx1[0] * factor + num_pad, bx1[1] * factor,
                           bx1[2] * factor + num_pad, bx1[3] * factor])
        bx2_rs = np.array([bx2[0] * factor + num_pad, bx2[1] * factor,
                           bx2[2] * factor + num_pad, bx2[3] * factor])
    else:
        factor = np.float(length) / np.float(w)
        num_pad = int(round((length - factor * h) / 2.0))
        bx1_rs = np.array([bx1[0] * factor, bx1[1] * factor + num_pad,
                           bx1[2] * factor, bx1[3] * factor + num_pad])
        bx2_rs = np.array([bx2[0] * factor, bx2[1] * factor + num_pad,
                           bx2[2] * factor, bx2[3] * factor + num_pad])

    # generate map
    map_1 = np.zeros([length, length], dtype=np.uint8)
    map_2 = np.zeros([length, length], dtype=np.uint8)
    # print('ff',int(round(bx1_rs[1])))
    for i in range(int(round(bx1_rs[1])), int(round(bx1_rs[3]))):
        for j in range(int(round(bx1_rs[0])), int(round(bx1_rs[2]))):
            map_1[i][j] = 1  # human的

    for i in range(int(round(bx2_rs[1])), int(round(bx2_rs[3]))):
        for j in range(int(round(bx2_rs[0])), int(round(bx2_rs[2]))):
            map_2[i][j] = 1  # object的
    return np.array([map_1, map_2]), bx1_rs, bx2_rs


def get_mean_and_std(bat_img):

    '''
    Compute the mean and std value of dataset.
    '''

    mean = np.zeros(3)
    var = np.zeros(3)
    for inputs in bat_img:
        for i in range(3):
            mean[i] += inputs[:, :, i].mean()
            var[i] += inputs[:, :, i].var()
    mean = mean / bat_img.shape[0]
    var = var / bat_img.shape[0]
    return mean, var


def to_one_hot(action_list, class_num):
    one_hot = np.zeros(class_num)
    if isinstance(action_list, list):
        for i in range(class_num):
            if i  in action_list:
                one_hot[i] = 1
            else:
                one_hot[i] = 0
    else:
        one_hot[action_list] = 1

    return one_hot


def batch_generator(x, pad, bs, _shuffle=False, classes_num=0, HICO_IMAGE_DIR=None, HICO_POINT_MASK_DIR=None):
    """
    returns:
    x_batch:
    y_batch:
    """
    while (True):
        if (_shuffle):
            new_ind = shuffle(range(x.shape[0]))  # random_state=155

            x = x.take(new_ind)

        bat_img, bat_roi_o, bat_roi_h, bat_pair_posi, bat_score_o, bat_img_path, bat_human_bbox, bat_obj_bbox, point_arr ,bat_olabel= [], [], [], [], [], [], [], [], [],[]
        bat_label = []

        for i in range(len(x)):
            row = x.iloc[i]

            # image
            img_path = os.path.join(HICO_IMAGE_DIR, row['name'])
            point_mask_path = os.path.join(HICO_POINT_MASK_DIR, str(i).zfill(7) + "_mask.jpg")
            bat_img_path.append(row['name'])
            if os.path.exists(img_path):

                img = Image.open(img_path)

                img_arr = np.array(img.resize((224, 224)))

                if len(img_arr.shape) == 2:
                    im3 = im2 = im1 = img.convert('L')
                    pic = Image.merge('RGB', (im1, im2, im3))
                    img_arr = np.array(pic.resize((224, 224)))

                # cv2.imwrite('./kk/{}.jpg'.format(str(i)),img_arr[..., ::-1])
            else:
                print("error", img_path)
            # if os.path.exists(point_mask_path):
            #
            #     point_mask = cv2.imread(point_mask_path)#bgr
            #     point_mask = cv2.resize(point_mask, (128, 128))
            #
            #     # img_arr= img[..., ::-1]# #rgb [::-1代表反序读]
            #
            # else:
            #     print("error",point_mask_path)

            bat_img.append(img_arr)
            bat_roi_h.append(row['h_44fm_roi_y1x1y2x2'])
            bat_roi_o.append(row['o_44fm_roi_y1x1y2x2'])

            bat_score_o.append(row['score'])
            bat_olabel.append(row['olabel'])

            label_one_hot = to_one_hot(row['action_no'], classes_num)
            bat_label.append(label_one_hot)

            # pairwise stream data - with padding & without padding
            hum_bbox = row['human_bbox']
            obj_bbox = row['obj_bbox']

            hum_bbox_t = [hum_bbox[0], hum_bbox[1], hum_bbox[2], hum_bbox[3]]
            obj_bbox_t = [obj_bbox[0], obj_bbox[1], obj_bbox[2], obj_bbox[3]]

            bat_human_bbox.append(hum_bbox)
            bat_obj_bbox.append(obj_bbox)

            if pad == 0:
                pair_posi = get_map_n_pad(hum_bbox_t, obj_bbox_t, 64)
            elif pad == 1:
                pair_posi = get_map_w_pad(hum_bbox_t, obj_bbox_t, 64)

            pair_posi = pair_posi[0]
            pair_posi = pair_posi.swapaxes(0, 1)
            pair_posi = pair_posi.swapaxes(1, 2)
            bat_pair_posi.append(pair_posi)
            # print("bat_pai_posi",np.array(bat_pair_posi).shape)

            if ((i + 1) % bs == 0):
                # first, to make the label_map
                # label_copy = sorted(set(bat_label[:]))
                #
                # if max(label_copy)>classes_num:
                #     label_map = {
                #         v: i + 1
                #         for i, v in enumerate(label_copy)
                #     }
                #     for i in range(len(bat_label)):
                #         bat_label[i] = label_map[bat_label[i]]
                mean, var = get_mean_and_std(np.array(bat_img) / 255)

                bat_index = np.array(range(0, bs))

                bat_roi_h = np.insert(np.array(bat_roi_h), 0, values=bat_index, axis=1)
                bat_roi_o = np.insert(np.array(bat_roi_o), 0, values=bat_index, axis=1)

                x_batch = [(np.array(bat_img) / 255 - mean) / var,  # 这里除了255
                           bat_roi_h,
                           bat_roi_o,
                           np.array(bat_pair_posi),
                           np.array(bat_score_o),
                           bat_img_path,
                           bat_human_bbox,
                           bat_obj_bbox,
                           point_arr,
                           bat_olabel]

                y_batch = np.array(bat_label)

                bat_img, bat_roi_o, bat_roi_h, bat_pair_posi, bat_score_o, bat_img_path, bat_human_bbox, bat_obj_bbox, point_arr ,bat_olabel= [], [], [], [], [], [], [], [], [], []
                bat_label = []

                yield (x_batch, y_batch)
            elif i + 1 == len(x):
                print('the last one')
                bs_copy = len(x) % bs
                mean, var = get_mean_and_std(np.array(bat_img) / 255)

                bat_index = np.array(range(0, bs_copy))

                bat_roi_h = np.insert(np.array(bat_roi_h), 0, values=bat_index, axis=1)
                bat_roi_o = np.insert(np.array(bat_roi_o), 0, values=bat_index, axis=1)

                x_batch = [(np.array(bat_img) / 255 - mean) / var,  # 这里除了255
                           bat_roi_h,
                           bat_roi_o,
                           np.array(bat_pair_posi),
                           np.array(bat_score_o),
                           bat_img_path,
                           bat_human_bbox,
                           bat_obj_bbox,
                           point_arr,
                           bat_olabel]

                y_batch = np.array(bat_label)

                bat_img, bat_roi_o, bat_roi_h, bat_pair_posi, bat_score_o, bat_img_path, bat_human_bbox, bat_obj_bbox, point_arr,bat_olabel = [], [], [], [], [], [], [], [], [],[]
                bat_label = []

                yield (x_batch, y_batch)


def batch_multi_generator(x, pad, bs, _shuffle=False, classes_num=0, IMAGE_DIR=None, POINT_MASK_DIR=None):
    """
    returns:
    x_batch:
    y_batch:
    """
    while (True):
        if (_shuffle):
            new_ind = shuffle(range(x.shape[0]))  # random_state=155

            x = x.take(new_ind)

        bat_img, bat_roi_o, bat_roi_h, bat_pair_posi, bat_score_o, bat_img_path, bat_human_bbox, bat_obj_bbox, bat_point = [], [], [], [], [], [], [], [], []
        bat_label = []

        for i in range(len(x)):
            row = x.iloc[i]

            # image
            img_path = os.path.join(IMAGE_DIR, row['name'])
            point_mask_path = os.path.join(POINT_MASK_DIR, str(i).zfill(7) + "_mask.jpg")
            bat_img_path.append(row['name'])
            if os.path.exists(img_path):

                img = Image.open(img_path)

                img_arr = np.array(img.resize((224, 224)))

                if len(img_arr.shape) == 2:
                    im3 = im2 = im1 = img.convert('L')
                    pic = Image.merge('RGB', (im1, im2, im3))
                    img_arr = np.array(pic.resize((224, 224)))

                # cv2.imwrite('./kk/{}.jpg'.format(str(i)),img_arr[..., ::-1])
            else:
                # print("error", img_path)
                pass
            if os.path.exists(point_mask_path):


                point_mask=Image.open(point_mask_path)

                point_mask_arr = np.array(point_mask.resize((128, 128)))

                if len(point_mask_arr.shape) == 2:
                    im3 = im2 = im1 = point_mask.convert('L')
                    pic = Image.merge('RGB', (im1, im2, im3))
                    point_mask_arr = np.array(pic.resize((128, 128)))

            else:
                pass

            bat_point.append(point_mask_arr)
            bat_img.append(img_arr)
            bat_roi_h.append(row['h_44fm_roi_y1x1y2x2'])
            bat_roi_o.append(row['o_44fm_roi_y1x1y2x2'])

            bat_score_o.append(row['score'])

            label_one_hot = to_one_hot(row['action_no'], classes_num)
            bat_label.append(label_one_hot)

            # pairwise stream data - with padding & without padding
            hum_bbox = row['human_bbox']#
            obj_bbox = row['obj_bbox']


            hum_bbox_t = [hum_bbox[0], hum_bbox[1], hum_bbox[2], hum_bbox[3]]
            obj_bbox_t = [obj_bbox[0], obj_bbox[1], obj_bbox[2], obj_bbox[3]]

            bat_human_bbox.append(hum_bbox)
            bat_obj_bbox.append(obj_bbox)

            if pad == 0:
                pair_posi = get_map_n_pad(hum_bbox_t, obj_bbox_t, 64)
            elif pad == 1:
                pair_posi = get_map_w_pad(hum_bbox_t, obj_bbox_t, 64)

            pair_posi = pair_posi[0]
            pair_posi = pair_posi.swapaxes(0, 1)
            pair_posi = pair_posi.swapaxes(1, 2)
            bat_pair_posi.append(pair_posi)
            # print("bat_pai_posi",np.array(bat_pair_posi).shape)

            if (i + 1) % bs == 0:
                # first, to make the label_map
                # label_copy = sorted(set(bat_label[:]))
                #
                # if max(label_copy) > classes_num:
                #     label_map = {
                #         v: i + 1
                #         for i, v in enumerate(label_copy)
                #     }
                #     for i in range(len(bat_label)):
                #         bat_label[i] = label_map[bat_label[i]]
                mean, var = get_mean_and_std(np.array(bat_img) / 255)

                bat_index = np.array(range(0, bs))

                bat_roi_h = np.insert(np.array(bat_roi_h), 0, values=bat_index, axis=1)
                bat_roi_o = np.insert(np.array(bat_roi_o), 0, values=bat_index, axis=1)

                x_batch = [(np.array(bat_img) / 255 - mean) / var,  # 这里除了255
                           bat_roi_h,
                           bat_roi_o,
                           np.array(bat_pair_posi),
                           np.array(bat_score_o),
                           bat_img_path,
                           bat_human_bbox,
                           bat_obj_bbox,
                           np.array( bat_point)/255]

                y_batch = np.array(bat_label)

                bat_img, bat_roi_o, bat_roi_h, bat_pair_posi, bat_score_o, bat_img_path, bat_human_bbox, bat_obj_bbox, bat_point = [], [], [], [], [], [], [], [], []
                bat_label = []

                yield (x_batch, y_batch)
            elif i + 1 == len(x):
                print('the last one')
                bs_copy = len(x) % bs
                mean, var = get_mean_and_std(np.array(bat_img) / 255)

                bat_index = np.array(range(0, bs_copy))

                bat_roi_h = np.insert(np.array(bat_roi_h), 0, values=bat_index, axis=1)
                bat_roi_o = np.insert(np.array(bat_roi_o), 0, values=bat_index, axis=1)

                x_batch = [(np.array(bat_img) / 255 - mean) / var,  # 这里除了255
                           bat_roi_h,
                           bat_roi_o,
                           np.array(bat_pair_posi),
                           np.array(bat_score_o),
                           bat_img_path,
                           bat_human_bbox,
                           bat_obj_bbox,
                           np.array(bat_point)/255]

                y_batch = np.array(bat_label)

                bat_img, bat_roi_o, bat_roi_h, bat_pair_posi, bat_score_o, bat_img_path, bat_human_bbox, bat_obj_bbox, bat_point = [], [], [], [], [], [], [], [], []
                bat_label = []

                yield (x_batch, y_batch)


if __name__ == '__main__':
    train_data = return_multi_data()
    HICO_IMAGE_DIR = "/home/priv-lab1/workspace/zxh/My_Database/hico_20160224_det/images/train2015"  # 训练测试时需要修改此处
    HICO_POINT_MASK_DIR = '/home/priv-lab1/workspace/zxh/pytorch/pytorch-openpose/results'
    CLASS_NUM = 600
    batch_size = 50

    batch_multi_generator(x=train_data, pad=1, bs=batch_size, HICO_IMAGE_DIR=HICO_IMAGE_DIR,
                          HICO_POINT_MASK_DIR=HICO_POINT_MASK_DIR, classes_num=CLASS_NUM)
