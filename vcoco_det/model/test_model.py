from datasets import return_data, batch_generator, return_multi_data, batch_multi_generator
import util
import test_generator as TS
import torch.nn as nn
import torch
from collections import OrderedDict
import os
import json
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend('agg')

DATASETS = 'multi_all'
CKPT_EPOCH = 50
HICO_IMAGE_DIR = "../DATA/MSCOCO2014/val2014"  # 训练测试时需要修改此处
HICO_POINT_MASK_DIR = '/home/priv-lab1/workspace/zxh/pytorch/pytorch-openpose/results'
CKPT_PATH = './results/weights/model_{}/model_{}.ckpt'.format(DATASETS, str(CKPT_EPOCH))
SAVE_RESULTS = './results/vcoco_det_results/test_{}_results.json'.format(DATASETS)
TEST_CSV_PATH = '/home/priv-lab1/workspace/zxh/HOI/DATA/v-coco/hoi-data/test/final_test_data.csv'
TXT_DIR = "./results/map/{}/predicted/".format(DATASETS)
MAP_DIR = './results/map/{}'.format(DATASETS)
NUM_CLASSES = 29
batch_size = 100
TOP_K = (1,)
results = OrderedDict()


def test():
    pass

    # load data
    test_data = return_data(TEST_CSV_PATH)
    print('test_data.shape', test_data.shape)
    gen = batch_generator(x=test_data, pad=1, bs=batch_size, classes_num=NUM_CLASSES, HICO_IMAGE_DIR=HICO_IMAGE_DIR,
                          HICO_POINT_MASK_DIR=HICO_POINT_MASK_DIR)
    batches = (test_data.shape[0]) // batch_size + 1  # 一个epoch训练整个数据集，这里的意思是在该batch_size下需要训练的batch的个数
    extra_batch = (test_data.shape[0]) % batch_size
    result_np = np.zeros((test_data.shape[0], NUM_CLASSES))
    label_np = np.zeros((test_data.shape[0], NUM_CLASSES))
    print("data loaded done")

    # build model

    model = torch.load(CKPT_PATH)
    print('model loadded done')

    # start testing
    acc_all = 0
    for batch_index in range(batches):
        batch = next(gen)

        batch = util.process_batch(batch)
        batch_imgs_tensor = batch[0]
        batch_rois_h_tensor = batch[1]
        batch_rois_o_tensor = batch[2]
        batch_pair_posi_tensor = batch[3]
        # batch_obj_det_s_tensor = batch[4]
        batch_img_path = batch[5]
        batch_human_bboxes = batch[6]
        batch_obj_bboxes = batch[7]
        batch_action_tensor = batch[8]
        batch_point_tensor= batch[9]
        batch_olabel=batch[10]
        #
        out = model(batch_imgs_tensor, batch_rois_h_tensor, batch_rois_o_tensor, batch_pair_posi_tensor)
        output_size = out.data.shape[0]
        # print(np.argsort(out.clone().cpu().detach().numpy())[:20])
        # print(batch_action_tensor.cpu().detach().numpy()[:20])

        # result_np[batch_index*batch_size:batch_index*batch_size+output_size,:]=out.clone().cpu().detach().numpy()
        # label_np[batch_index*batch_size:batch_index*batch_size+output_size,:]=batch_action_tensor.clone().cpu().detach().numpy()

        #
        # auc=roc_auc_score(batch_action_tensor.clone().cpu().detach().numpy()[:,i],out.clone().cpu().detach().numpy()[:,i])

        acc = util.multi_accuracy(out.clone(), batch_action_tensor, TOP_K)  # top n+1
        acc_all = acc[0].item() / 100 + acc_all

        predicteds_list, score_list = TS.get_predicteds(out, TOP_K)
        print("batch {}/{}".format(str(batch_index+1), str(batches)), '----', 'acc', acc[0].item() / 100)

        for i in range(output_size):
            results[i + batch_index * batch_size] = {'img_path': batch_img_path[i], 'hbbox': batch_human_bboxes[i],
                                                     'obbox': batch_obj_bboxes[i],
                                                     'predicteds': predicteds_list[i].tolist(),
                                                     'scores': score_list[i].tolist(),'olabel':batch_olabel[i]}

    json.dump(results, open(SAVE_RESULTS, 'w'), indent=4)
    print(acc_all / batches)



def get_results_txt():
    with open(SAVE_RESULTS, 'r') as load_f:
        load_dict = json.load(load_f)
    TS.make_predicted_txt(load_dict, TXT_DIR)


if __name__ == '__main__':
    test()
    get_results_txt()
    # map.hico_map(MAP_DIR)
    # compute_map()
