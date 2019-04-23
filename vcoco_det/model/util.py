import torch
import os
import math
import time
import sys
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def get_box_index(batch_size):
    batch_index_list = [i for i in range(batch_size)]
    box_index_data = torch.IntTensor(batch_index_list)
    box_index = to_varabile(box_index_data)  # ([0,1,2,3....,batch-1])
    return box_index


def to_varabile(tensor, is_cuda=True):
    if is_cuda:
        tensor = tensor.cuda()
    return tensor


def process_multi_batch(batch):
    """
    :param batch:
    :return:
    batch[0]:image_tensor
    batch[1]:rois_h_tensor
    batch[2]:rois_o_tensor
    batch[3]:pair_tensor
    batch[4]:obj_det_score_tensor
    batch[5]:img_path
    batch[6]:human_bboxes
    batch[7]:obj_bboxes
    batch[8]:action_tensor
    batch[9]:batch_point_arr_tensor
    """
    batch_imgs_arr = batch[0][0].transpose(0, 3, 2, 1)
    batch_rois_h_arr = batch[0][1]
    batch_rois_o_arr = batch[0][2]
    batch_pair_posi_arr = batch[0][3].transpose(0, 3, 2, 1)
    batch_obj_det_s_arr = batch[0][4]
    batch_img_path = batch[0][5]
    batch_human_bboxes = batch[0][6]
    batch_obj_bboxes = batch[0][7]
    batch_point_arr = batch[0][8].transpose(0, 3, 2, 1)


    # label
    batch_action_arr = batch[1]

    batch_imgs_tensor = torch.from_numpy(batch_imgs_arr)
    batch_imgs_tensor = to_varabile(batch_imgs_tensor.float())

    batch_rois_h_arr_tensor = torch.from_numpy(batch_rois_h_arr)
    batch_rois_h_arr_tensor = to_varabile(batch_rois_h_arr_tensor.float())

    batch_rois_o_arr_tensor = torch.from_numpy(batch_rois_o_arr)
    batch_rois_o_arr_tensor = to_varabile(batch_rois_o_arr_tensor.float())

    batch_pair_posi_arr_tensor = torch.from_numpy(batch_pair_posi_arr)
    batch_pair_posi_arr_tensor = to_varabile(batch_pair_posi_arr_tensor.float())

    batch_obj_det_s_arr_tensor = torch.from_numpy(batch_obj_det_s_arr)
    batch_obj_det_s_arr_tensor = to_varabile(batch_obj_det_s_arr_tensor.float())
    #
    batch_point_arr_tensor = torch.from_numpy(batch_point_arr)
    batch_point_arr_tensor = to_varabile(batch_point_arr_tensor.float())

    batch_action_tensor = torch.from_numpy(batch_action_arr).type(torch.FloatTensor)  # .type(torch.FloatTensor)
    batch_action_tensor = to_varabile(batch_action_tensor)

    return batch_imgs_tensor, batch_rois_h_arr_tensor, batch_rois_o_arr_tensor, batch_pair_posi_arr_tensor, \
           batch_obj_det_s_arr_tensor, batch_img_path, batch_human_bboxes, batch_obj_bboxes, \
           batch_action_tensor ,  batch_point_arr_tensor


def process_batch(batch):
    """
    :param batch:
    :return:
    batch[0]:image_tensor
    batch[1]:rois_h_tensor
    batch[2]:rois_o_tensor
    batch[3]:pair_tensor
    batch[4]:obj_det_score_tensor
    batch[5]:img_path
    batch[6]:human_bboxes
    batch[7]:obj_bboxes
    batch[8]:action_tensor
    """
    batch_imgs_arr = batch[0][0].transpose(0, 3, 2, 1)
    batch_rois_h_arr = batch[0][1]
    batch_rois_o_arr = batch[0][2]
    batch_pair_posi_arr = batch[0][3].transpose(0, 3, 2, 1)
    batch_obj_det_s_arr = batch[0][4]
    batch_img_path = batch[0][5]
    batch_human_bboxes = batch[0][6]
    batch_obj_bboxes = batch[0][7]
    # batch_point_arr = batch[0][8].transpose(0, 3, 2, 1)
    batch_olabel = batch[0][9]

    # label
    batch_action_arr = batch[1]

    batch_imgs_tensor = torch.from_numpy(batch_imgs_arr)
    batch_imgs_tensor = to_varabile(batch_imgs_tensor.float())

    batch_rois_h_arr_tensor = torch.from_numpy(batch_rois_h_arr)
    batch_rois_h_arr_tensor = to_varabile(batch_rois_h_arr_tensor.float())

    batch_rois_o_arr_tensor = torch.from_numpy(batch_rois_o_arr)
    batch_rois_o_arr_tensor = to_varabile(batch_rois_o_arr_tensor.float())

    batch_pair_posi_arr_tensor = torch.from_numpy(batch_pair_posi_arr)
    batch_pair_posi_arr_tensor = to_varabile(batch_pair_posi_arr_tensor.float())

    batch_obj_det_s_arr_tensor = torch.from_numpy(batch_obj_det_s_arr)
    batch_obj_det_s_arr_tensor = to_varabile(batch_obj_det_s_arr_tensor.float())
    #
    # batch_point_arr_tensor = torch.from_numpy(batch_point_arr)
    # batch_point_arr_tensor = to_varabile(batch_point_arr_tensor)

    batch_action_tensor = torch.from_numpy(batch_action_arr)
    batch_action_tensor = to_varabile(batch_action_tensor)

    return batch_imgs_tensor, batch_rois_h_arr_tensor, batch_rois_o_arr_tensor, batch_pair_posi_arr_tensor, \
           batch_obj_det_s_arr_tensor, batch_img_path, batch_human_bboxes, batch_obj_bboxes, \
           batch_action_tensor ,' ',batch_olabel


class FocalLoss(nn.Module):

    def __init__(self, focusing_param=8, balance_param=0.25):  # focusing_param代表gamma
        super(FocalLoss, self).__init__()

        self.focusing_param = focusing_param
        self.balance_param = balance_param

    def forward(self, output, target):
        cross_entropy = F.cross_entropy(output, target)
        cross_entropy_log = torch.log(cross_entropy)
        logpt = - F.cross_entropy(output, target)
        pt = torch.exp(logpt)

        focal_loss = -((1 - pt) ** self.focusing_param) * logpt

        balanced_focal_loss = self.balance_param * focal_loss

        return balanced_focal_loss


def accuracy(output, target, topk=(5,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def from_one_hot_to_list(one_hot_tensor):
    one_hot_list = one_hot_tensor.clone().cpu().detach().numpy().tolist()

    list_ = []
    for one_hot in one_hot_list:
        list_.append(one_hot.index(1.0))
    return np.array(list_)


def multi_accuracy(output, target, topk=(5,)):
    maxk = max(topk)

    target = torch.from_numpy(from_one_hot_to_list(target)).cuda()
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.reset()

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

    def reset(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def adjust_learning_rate(ori_lr, optimizer, epoch, batch_index, lr_decay_step):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    rate = epoch // lr_decay_step
    urate = epoch % lr_decay_step

    if epoch > 1 and urate == 0 and batch_index == 1:
        lr = ori_lr * (0.1 ** rate)
        optimizer.param_groups[0]['lr'] = lr


def get_loss_weighted():
    hoi_class_ins = [1838,2150,935,252,300,1770,164,301,319,243,167,236,159,152,127,305,210,190,239,220,678,63,65,15,51,176]
    hoi_num = 0
    for i in hoi_class_ins:
        hoi_num += i
    hoi_class_weight = [hoi_num / float(ins) for ins in hoi_class_ins]
    return torch.from_numpy(np.array(hoi_class_weight))

    # assert len(action_classes) == len(hoi_class_ins)
    # hoi_class_weight = [2000.0 / float(ins) for ins in hoi_class_ins]

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
    weight = get_loss_weighted()
    # print(weight)
