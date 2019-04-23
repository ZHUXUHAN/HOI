from collections import OrderedDict
import torch
import os
import numpy as np
def get_predicteds(output,topk=(5,)):
    """
    :param output: model's output tensor
    :param topk: a tuple for topk (top_start,top_end)
    :return: preds_list scores_list
    """
    maxk = max(topk)
    scores, preds = output.topk(maxk, 1, True, True)
    return preds.cpu().detach().numpy(),scores.cpu().detach().numpy()
def multi_get_predicteds(output,score):
    """
    :param output: model's output tensor
    :param topk: a tuple for topk (top_start,top_end)
    :return: preds_list scores_list
    """
    preds = np.where(output>score)
    print(preds)
    scores= output[preds]
    # return preds.cpu().detach().numpy(),scores.cpu().detach().numpy()
def make_predicted_txt(predicted_dict,txt_dir):
    """
    just make predicted txt for every file in test, and
    calculate map
    predicted_dict:your predicted dict(maybe saved as a json file)
    :return:

    """

    human_bbox_dict = OrderedDict()
    object_bbox_dict = OrderedDict()
    action_dict = OrderedDict()
    scores_dict = OrderedDict()
    olabels_dict = OrderedDict()
    filenames = []


    for index, row in predicted_dict.items():
        if row['img_path'] in filenames:
            pass
        else:
            filenames.append(row['img_path'])
        if row['img_path'] in human_bbox_dict:
            human_bbox_dict[row['img_path']].append(row['hbbox'])
            object_bbox_dict[row['img_path']].append(row['obbox'])
            action_dict[row['img_path']].append(row['predicteds'])
            scores_dict[row['img_path']].append(row['scores'])
            olabels_dict[row['img_path']].append(row['olabel'])
        else:
            human_bbox_dict[row['img_path']] = [row['hbbox']]
            object_bbox_dict[row['img_path']] = [row['obbox']]
            action_dict[row['img_path']]= [row['predicteds']]
            scores_dict[row['img_path']]= [row['scores']]
            olabels_dict[row['img_path']]=[row['olabel']]
    print("data load done")

    for file in filenames:
        obboxs = object_bbox_dict[file]
        actions = action_dict[file]
        humans=human_bbox_dict[file]
        scores=scores_dict[file]
        olabels=olabels_dict[file]
        txtname=file.split('.')[0]+'.txt'
        save_lists=[]
        for i, hbbox in enumerate(humans):
            h_x1 = hbbox[0]
            h_x2 = hbbox[2]
            h_y1 = hbbox[1]
            h_y2 = hbbox[3]
            o_x1 = obboxs[i][0]
            o_x2 = obboxs[i][2]
            o_y1 = obboxs[i][1]
            o_y2 = obboxs[i][3]

            for ii,label in enumerate(actions[i]):
                label=label
                score=scores[i][ii]
                olabel=olabels[i]
                save_lists.append(str(label)+' '+str(score)+' '+str(h_x1)+' '+str(h_y1)+' '+str(h_x2)+' '+str(h_y2)+' '+str(o_x1)+' '+str(o_y1)+' '+str(o_x2)+' '+str(o_y2)+ ' '+str(olabel)+'\n')
        if not os.path.exists(txt_dir):
            os.makedirs(txt_dir)
        with open(txt_dir+txtname,'w') as f:
            for save_list in save_lists:
                f.write(save_list)
    print("save into txt")

def make_pkl():
    pass


