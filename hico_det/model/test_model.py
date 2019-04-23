import hico_model
from datasets import return_data, batch_generator,return_multi_data,batch_multi_generator
import util
import test_generator as TS
import torch.nn as nn
import torch
from collections import OrderedDict
import os
from lib import map
import json
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
plt.switch_backend('agg')

DATASETS = 'multi_all'
CKPT_EPOCH = 60
HICO_IMAGE_DIR = "/home/priv-lab1/workspace/zxh/My_Database/hico_20160224_det/images/test2015"  # 训练测试时需要修改此处
HICO_POINT_MASK_DIR = '/home/priv-lab1/workspace/zxh/pytorch/pytorch-openpose/results'
CKPT_PATH = '/home/priv-lab1/workspace/zxh/HOI/hico_det/results/weights/model_{}/model_{}.ckpt'.format(DATASETS,
                                                                                                       str(CKPT_EPOCH))
SAVE_RESULTS = './results/hico_det_results/test_{}_results.json'.format(DATASETS)
TEST_CSV_PATH = '/home/priv-lab1/workspace/zxh/HOI/hico_det/data/hico_data/final_test_data.csv'
TXT_DIR = "/home/priv-lab1/workspace/zxh/HOI/hico_det/results/map/{}/predicted/".format(DATASETS)
MAP_DIR = '/home/priv-lab1/workspace/zxh/HOI/hico_det/results/map/{}'.format(DATASETS)
NUM_CLASSES = 600
batch_size = 100
TOP_K = (5,)
results = OrderedDict()



def test():
    pass

    # load data
    test_data = return_data(TEST_CSV_PATH)
    print('test_data.shape', test_data.shape)
    gen = batch_generator(x=test_data, pad=1, bs=batch_size, classes_num=NUM_CLASSES, HICO_IMAGE_DIR=HICO_IMAGE_DIR,
                          HICO_POINT_MASK_DIR=HICO_POINT_MASK_DIR)
    batches = (test_data.shape[0]) // batch_size +1  # 一个epoch训练整个数据集，这里的意思是在该batch_size下需要训练的batch的个数
    extra_batch= (test_data.shape[0]) % batch_size
    result_np = np.zeros(( test_data.shape[0], 600))
    label_np = np.zeros(( test_data.shape[0], 600))
    print("data loaded done")

    # build model

    model = torch.load(CKPT_PATH)
    print('model loadded done')

    # start testing
    acc_all=0
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
    #
        out = model(batch_imgs_tensor, batch_rois_h_tensor, batch_rois_o_tensor, batch_pair_posi_tensor)
        output_size=out.data.shape[0]
        # print(np.argsort(out.clone().cpu().detach().numpy()))

        # result_np[batch_index*batch_size:batch_index*batch_size+output_size,:]=out.clone().cpu().detach().numpy()
        # label_np[batch_index*batch_size:batch_index*batch_size+output_size,:]=batch_action_tensor.clone().cpu().detach().numpy()

#
    # auc=roc_auc_score(batch_action_tensor.clone().cpu().detach().numpy()[:,i],out.clone().cpu().detach().numpy()[:,i])



        acc = util.multi_accuracy(out.clone(), batch_action_tensor, TOP_K)  # top n+1
        acc_all=acc[0].item() / 100+acc_all

        predicteds_list, score_list = TS.get_predicteds(out, TOP_K)
        print("batch {}/{}".format(str(batch_index),str(batches)), '----', 'acc', acc[0].item() / 100)

        for i in range(output_size):
            results[i + batch_index * batch_size] = {'img_path': batch_img_path[i], 'hbbox': batch_human_bboxes[i],
                                                         'obbox': batch_obj_bboxes[i],
                                                         'predicteds': predicteds_list[i].tolist(),
                                                         'scores': score_list[i].tolist()}

    json.dump(results, open(SAVE_RESULTS, 'w'), indent=4)
    print(acc_all/batches)
    # np.save('./results/hico_det_results/result_np.npy', result_np)
    # np.save('./results/hico_det_results/label_np.npy', label_np)



def get_results_txt():
    with open(SAVE_RESULTS, 'r') as load_f:
        load_dict = json.load(load_f)
    TS.make_predicted_txt(load_dict, TXT_DIR)
def compute_map():
    total_auc = 0
    precision = dict()
    recall = dict()
    average_precision = dict()
    result_np = np.load('./results/hico_det_results/result_np.npy')
    label_np = np.load('./results/hico_det_results/label_np.npy')
    #
    # for i in range(600):
    #     precision[i], recall[i], _ = precision_recall_curve( label_np[:, i],
    #                                                          result_np[:, i])
    #     average_precision[i] = average_precision_score(label_np[:, i], result_np[:, i])
    # #
    #
    #
    #     auc = roc_auc_score(label_np[:, i], result_np[:, i])
    #     total_auc = total_auc + auc
    #     print(i, auc)
    # print('the end map', total_auc / 600)
    precision["micro"], recall["micro"], _ = precision_recall_curve(label_np.ravel(),
                                                                    result_np.ravel())
    average_precision["micro"] = average_precision_score(label_np, result_np,
                                                         average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.4f}'
          .format(average_precision["micro"]))
    plt.figure()
    plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall["micro"], precision["micro"], alpha=0.2, color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Average precision score, micro-averaged over all classes: AP={0:0.4f}'.format(average_precision["micro"]))
    # plt.imshow()
    plt.savefig('im.png')

if __name__ == '__main__':
    test()
    get_results_txt()
    # map.hico_map(MAP_DIR)
    # compute_map()
