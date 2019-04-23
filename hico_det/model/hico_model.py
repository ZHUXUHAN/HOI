import torch.nn as nn
import torch
from torchvision import datasets,transforms,utils, models
# from roi_pooling.crop_and_resize import CropAndResizeFunction
from roi_pooling import RoIPoolFunction
from torch.autograd import Variable
import util
import torch.nn.functional as F
import numpy as np
NUM_CLASSES=600
FC_DIM=2048
class BackBone_Model(nn.Module):
    def __init__(self,net_name):
        super( BackBone_Model, self).__init__()
        if net_name=='vgg16':
            model = models.vgg16()
            model.load_state_dict(torch.load(
            '/home/priv-lab1/workspace/zxh/pytorch/model/vgg16-397923af.pth'))  # 只加载参数#map_location={'cuda:1': 'cuda:0'}
            self.model = model.features[:31]
        if net_name=='resnet50':
            model=models.resnet50()
            model.load_state_dict(torch.load(
                '/home/priv-lab1/workspace/zxh/pytorch/model/resnet50-19c8e357.pth'))  # 只加载参数#map_location={'cuda:1': 'cuda:0'}

            self.model = nn.Sequential(model.conv1, model.bn1, model.relu,
                                  model.maxpool, model.layer1, model.layer2, model.layer3, model.layer4)


    def forward(self, x):
        for name, layer in self.model._modules.items():
            x=layer(x)
        out=x
        return out
class Human_Stream(nn.Module):
    def __init__(self):
        super(Human_Stream, self).__init__()
        self.human_fc = nn.Sequential(
            nn.Linear(4*4*2048, FC_DIM),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(FC_DIM, FC_DIM),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(FC_DIM, NUM_CLASSES)
        )

    def forward(self, x):#输入通道维度为（512，4，4）

        x = x.view(-1,4*4*2048)
        out = self.human_fc(x)
        return out
class Pair_Stream(nn.Module):
    def __init__(self):
        super(Pair_Stream, self).__init__()
        self.pair_conv = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1),#输入的两个通道，输入前后特征图大小不变
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(stride=2, kernel_size=3),
            nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=1),#注意这里有个5的卷积核
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(stride=2, kernel_size=3),
    )
        self.pair_fc=nn.Sequential(
            nn.Linear(14*14*32,256),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(256,NUM_CLASSES)
        )

    def forward(self, x):  #输入通道维度为（2，64，64）

        x=self.pair_conv(x)
        x=x.view(-1,14*14*32)
        out=self.pair_fc(x)
        return out
class Object_Stream(nn.Module):
    def __init__(self):
        super(Object_Stream, self).__init__()
        self.object_fc = nn.Sequential(
            nn.Linear(4 * 4 * 2048, FC_DIM),#如果使用vgg16则将所有的4*4*2048中的2048改为512
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(FC_DIM, FC_DIM),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(FC_DIM, NUM_CLASSES)#注意类别要改的
        )

    def forward(self, x):  # 输入通道维度为（512，4，4）
        # print('x.shape',x.shape)

        x = x.view(-1, 4 * 4 * 2048)
        out = self.object_fc(x)
        return out


class ROI_Pooling(nn.Module):

    def __init__(self, crop_height, crop_width, spatial_scale=1.0):
        super(ROI_Pooling, self).__init__()

        self.crop_height = crop_height
        self.crop_width = crop_width
        self.spatial_scale = spatial_scale
    def forward(self, input, rois):
        size=(self.crop_height, self.crop_width)
        assert rois.dim() == 2
        assert rois.size(1) == 5
        output = []
        rois = rois.data.float()
        num_rois = rois.size(0)


        rois[:, 1:].mul_(self.spatial_scale)
        rois = rois.long()

        for i in range(num_rois):
            roi = rois[i]
            im_idx = roi[0]
            im = input.narrow(0, im_idx.data.item()%num_rois, 1)[..., roi[1]:(roi[3] + 1), roi[2]:(roi[4] + 1)]  # 取零维的 注意此处对应的是(y1,x1,y2,x2)
            output.append(F.adaptive_max_pool2d(im, size))
        output = torch.cat(output, 0)
        return output
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model= BackBone_Model('resnet50')
        self.human_stream_model=Human_Stream()
        self.object_stream_model=Object_Stream()
        self.pair_stream_model=Pair_Stream()
        # self.box_index=util.get_box_index(batch_size)
        self.roi_pooling=ROI_Pooling(4,4,1.0)

        # self.score_stream_model=ObjScore_Stream()

    def forward(self, img_input,roi_h,roi_o,pair_input):
        feature_outputs = self.model(img_input)

        # print('size',feature_outputs.data.size())
        crops_h_tensor = self.roi_pooling(feature_outputs,roi_h)
        crops_o_tensor = self.roi_pooling(feature_outputs,roi_o)
        # print(crops_h_tensor)


        # print('crops_o_tensor_size', crops_o_tensor.data.size())
        human_stream_outputs = self.human_stream_model(crops_h_tensor)
        # print('human_stream_outputs.size',human_stream_outputs.data.size())
        object_stream_outputs = self.object_stream_model(crops_o_tensor)
        # print('object_stream_outputs.size', object_stream_outputs.data.size())

        pair_stream_outputs = self.pair_stream_model(pair_input)
        out=human_stream_outputs+object_stream_outputs+pair_stream_outputs
        # print("human_stream_outputs",human_stream_outputs)
        # print("object_stream_outputs",object_stream_outputs)
        # print("pair_stream_outputs",pair_stream_outputs)

        out=F.sigmoid(out)

        # print(out.shape)

        return out


if __name__ == '__main__':
    model=Model(10)
    print(model)
