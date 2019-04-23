import hico_model
from datasets import return_data,batch_generator,return_multi_data,batch_multi_generator
import util
import torch.nn as nn
import torch
import os
import random
import torch.nn.functional as F
import time
import torch.backends.cudnn as cudnn
from collections import defaultdict
HICO_IMAGE_DIR = "/home/priv-lab1/workspace/zxh/My_Database/hico_20160224_det/images/train2015"#训练测试时需要修改此处
HICO_POINT_MASK_DIR ='/home/priv-lab1/workspace/zxh/pytorch/pytorch-openpose/results'
CSV_FILE='/home/priv-lab1/workspace/zxh/HOI/hico_det/data/multi_label/train/final_multi_train_data.csv'
DATASETS='multi_all'
SAVE_CKPT_DIR='./results/weights/model_{}'.format(DATASETS)
device_ids=[0,2]
batch_size= 200  #130#单卡的50就够了 双卡不要超过200 130：9067个batches 必须是gpus的整数倍 要不在roi-pooling部分会出错
epoches = 30
ckpt_epoch =30
TOK_K =(1,)
Learning_Rate=0.01
LR_DECAY_STEP=60
ADJUST_LR=True
CLASS_NUM=600
RESUME_CKPT_PATH = '/home/priv-lab1/workspace/zxh/HOI/hico_det/results/weights/model_multi_all/model_60.ckpt'
RESUME_MODEL =True
def main():

    #load data
    train_data = return_multi_data(CSV_FILE)
    print('train_data.shape', train_data.shape)
    gen = batch_multi_generator(x=train_data, pad=1, bs=batch_size,HICO_IMAGE_DIR=HICO_IMAGE_DIR,HICO_POINT_MASK_DIR=HICO_POINT_MASK_DIR,classes_num=CLASS_NUM)
    batches = (train_data.shape[0]) // batch_size +1 # 一个epoch训练整个数据集，这里的意思是在该batch_size下需要训练的batch的个数
    print("data loaded done")

    #build model
    if RESUME_MODEL:
        model = torch.load(RESUME_CKPT_PATH)

    else:
        model= hico_model.Model()
        model = model.cuda()
        cudnn.benchmark = True
        model = nn.DataParallel(model, device_ids=device_ids)
    print(model)

    print('model builded done')

    #loss
    crossEntropyLoss = nn.CrossEntropyLoss()
    becloss =nn.BCELoss()
    weighted_becloss=nn.BCELoss(weight=util.get_loss_weighted().type(torch.cuda.FloatTensor))
    focalloss = util.FocalLoss()
    # optimizer
    optimizer_model = torch.optim.SGD(model.parameters(), lr=Learning_Rate, momentum=0.9, weight_decay=0.0005)


    #start training
    model.train()
    timers = defaultdict(util.Timer)
    data_num=batches*epoches


    for epoch in range(epoches):
        for batch_index in range(batches):
            timers['batch'].tic()
            batch = next(gen)
            batch=util.process_multi_batch(batch)
            batch_imgs_tensor=batch[0]
            batch_rois_h_tensor=batch[1]
            batch_rois_o_tensor=batch[2]
            batch_pair_posi_tensor=batch[3]
            batch_action_tensor=batch[8]


            out = model(batch_imgs_tensor, batch_rois_h_tensor, batch_rois_o_tensor, batch_pair_posi_tensor)

            loss =  weighted_becloss(out, batch_action_tensor)
            acc = util.multi_accuracy(out.clone(), batch_action_tensor,TOK_K)  # top n+1
            # print(out.clone().topk(1, 1, True, True)[0].t())

            #sgd
            optimizer_model.zero_grad()
            loss.backward()
            optimizer_model.step()
            if ADJUST_LR:
                util.adjust_learning_rate(Learning_Rate,optimizer_model, epoch+1,batch_index+1,LR_DECAY_STEP)
            lr=optimizer_model.param_groups[0]['lr']
            #time
            timers['batch'].toc()

            one_batch_time = timers['batch'].average_time
            elapsed_time=one_batch_time*(batch_index+batches*epoch+1) #注意这个地方很关键
            all_time = one_batch_time*data_num
            need_time=all_time-elapsed_time

            need_time=util.format_time(need_time)
            one_batch_time=util.format_time(one_batch_time)

            #train record

            # print("epoch:{}/{}".format(epoch+1,epoches), "--", "batch:{}/{}".format(str(batch_index+1),str(batches)),'--','lr',round(lr,8), "--", "loss", round(loss.data[0].item(),8),'--', 'acc',
            #       str(round(acc[0].item()/100,4)) , '--', 'time : {}'.format(one_batch_time),'--','need-time: {}'.format(need_time))
            print("epoch:{}/{}".format(epoch + 1, epoches), "--",
                  "batch:{}/{}".format(str(batch_index + 1), str(batches)), '--', 'lr', round(lr, 8), "--", "loss",
                  round(loss.data[0].item(), 8), '--', 'acc',
                  str(0), '--', 'time : {}'.format(one_batch_time), '--',
                  'need-time: {}'.format(need_time))

        if (epoch + 1) % ckpt_epoch == 0:
            if not os.path.exists(SAVE_CKPT_DIR):
                os.makedirs(SAVE_CKPT_DIR)
            torch.save(model, os.path.join(SAVE_CKPT_DIR,'model_{}.ckpt'.format(str(epoch+1+60))))



if __name__ == '__main__':
    main()
