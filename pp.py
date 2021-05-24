from __future__ import print_function
from itertools import product as product
from math import ceil
import torch.nn as nn
import torchvision.models._utils as _utils
import math
from torchvision import models
import torch.utils.model_zoo as model_zoo
import torch.backends.cudnn as cudnn
import queue as Queue
import os
import cv2
import sys
import collections
import torch
import argparse
import numpy as np
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
'''*********************************************face************************************************'''

#FaceNets/layer.py
# ---------------------------------------------------#
#   卷积块
#   Conv2D + BatchNormalization + LeakyReLU
# ---------------------------------------------------#
def facelayers_conv_bn(inp, oup, stride=1, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )


def facelayers_conv_bn1X1(inp, oup, stride, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )


# ---------------------------------------------------#
#   卷积块
#   Conv2D + BatchNormalization
# ---------------------------------------------------#
def facelayers_conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )


# ---------------------------------------------------#
#   多尺度加强感受野
# ---------------------------------------------------#
class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if (out_channel <= 64):
            leaky = 0.1

        # 3x3卷积
        self.conv3X3 = facelayers_conv_bn_no_relu(in_channel, out_channel // 2, stride=1)

        # 利用两个3x3卷积替代5x5卷积
        self.conv5X5_1 = facelayers_conv_bn(in_channel, out_channel // 4, stride=1, leaky=leaky)
        self.conv5X5_2 = facelayers_conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)

        # 利用三个3x3卷积替代7x7卷积
        self.conv7X7_2 = facelayers_conv_bn(out_channel // 4, out_channel // 4, stride=1, leaky=leaky)
        self.conv7x7_3 = facelayers_conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)

    def forward(self, inputs):
        conv3X3 = self.conv3X3(inputs)

        conv5X5_1 = self.conv5X5_1(inputs)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        # 所有结果堆叠起来
        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1
        self.output1 = facelayers_conv_bn1X1(in_channels_list[0], out_channels, stride=1, leaky=leaky)
        self.output2 = facelayers_conv_bn1X1(in_channels_list[1], out_channels, stride=1, leaky=leaky)
        self.output3 = facelayers_conv_bn1X1(in_channels_list[2], out_channels, stride=1, leaky=leaky)

        self.merge1 = facelayers_conv_bn(out_channels, out_channels, leaky=leaky)
        self.merge2 = facelayers_conv_bn(out_channels, out_channels, leaky=leaky)

    def forward(self, inputs):
        # -------------------------------------------#
        #   获得三个shape的有效特征层
        #   分别是C3  80, 80, 64
        #         C4  40, 40, 128
        #         C5  20, 20, 256
        # -------------------------------------------#
        inputs = list(inputs.values())

        # -------------------------------------------#
        #   获得三个shape的有效特征层
        #   分别是output1  80, 80, 64
        #         output2  40, 40, 64
        #         output3  20, 20, 64
        # -------------------------------------------#
        output1 = self.output1(inputs[0])
        output2 = self.output2(inputs[1])
        output3 = self.output3(inputs[2])

        # -------------------------------------------#
        #   output3上采样和output2特征融合
        #   output2  40, 40, 64
        # -------------------------------------------#
        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        # -------------------------------------------#
        #   output2上采样和output1特征融合
        #   output1  80, 80, 64
        # -------------------------------------------#
        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        out = [output1, output2, output3]
        return out
        
 
#FaceNets/mobilenet025.py
def facemobilenet_conv_bn(inp, oup, stride=1, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )


def facemobilenet_conv_dw(inp, oup, stride=1, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            # 640,640,3 -> 320,320,8
            facemobilenet_conv_bn(3, 8, 2, leaky=0.1),
            # 320,320,8 -> 320,320,16
            facemobilenet_conv_dw(8, 16, 1),

            # 320,320,16 -> 160,160,32
            facemobilenet_conv_dw(16, 32, 2),
            facemobilenet_conv_dw(32, 32, 1),

            # 160,160,32 -> 80,80,64
            facemobilenet_conv_dw(32, 64, 2),
            facemobilenet_conv_dw(64, 64, 1),
        )
        # 80,80,64 -> 40,40,128
        self.stage2 = nn.Sequential(
            facemobilenet_conv_dw(64, 128, 2),
            facemobilenet_conv_dw(128, 128, 1),
            facemobilenet_conv_dw(128, 128, 1),
            facemobilenet_conv_dw(128, 128, 1),
            facemobilenet_conv_dw(128, 128, 1),
            facemobilenet_conv_dw(128, 128, 1),
        )
        # 40,40,128 -> 20,20,256
        self.stage3 = nn.Sequential(
            facemobilenet_conv_dw(128, 256, 2),
            facemobilenet_conv_dw(256, 256, 1),
        )
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 1000)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        # x = self.model(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x
       
#FaceNets/retinaface.py
# ---------------------------------------------------#
#   种类预测（是否包含人脸）
# ---------------------------------------------------#
class faceretina_ClassHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=2):
        super(faceretina_ClassHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels, self.num_anchors * 2, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 2)


# ---------------------------------------------------#
#   预测框预测
# ---------------------------------------------------#
class faceretina_BboxHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=2):
        super(faceretina_BboxHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 4, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 4)


# ---------------------------------------------------#
#   人脸关键点预测
# ---------------------------------------------------#
class faceretina_LandmarkHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=2):
        super(faceretina_LandmarkHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 10, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 10)


class faceretina_RetinaFace(nn.Module):
    def __init__(self, cfg=None, pretrained=False, mode='train'):
        super(faceretina_RetinaFace, self).__init__()
        backbone = None
        if cfg['name'] == 'mobilenet0.25':
            backbone = MobileNetV1()
            if pretrained:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                checkpoint = torch.load("./model_data/mobilenetV1X0.25_pretrain.tar", map_location=device)
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:]
                    new_state_dict[name] = v
                backbone.load_state_dict(new_state_dict)
        elif cfg['name'] == 'Resnet50':
            backbone = models.resnet50(pretrained=pretrained)

        self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])

        # 获得每个初步有效特征层的通道数
        in_channels_list = [cfg['in_channel'] * 2, cfg['in_channel'] * 4, cfg['in_channel'] * 8]
        self.fpn = FPN(in_channels_list, cfg['out_channel'])
        # 利用ssh模块提高模型感受野
        self.ssh1 = SSH(cfg['out_channel'], cfg['out_channel'])
        self.ssh2 = SSH(cfg['out_channel'], cfg['out_channel'])
        self.ssh3 = SSH(cfg['out_channel'], cfg['out_channel'])

        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=cfg['out_channel'])

        self.mode = mode

    def _make_class_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(faceretina_ClassHead(inchannels, anchor_num))
        return classhead

    def _make_bbox_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(faceretina_BboxHead(inchannels, anchor_num))
        return bboxhead

    def _make_landmark_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(faceretina_LandmarkHead(inchannels, anchor_num))
        return landmarkhead

    def forward(self, inputs):
        # -------------------------------------------#
        #   获得三个shape的有效特征层
        #   分别是C3  80, 80, 64
        #         C4  40, 40, 128
        #         C5  20, 20, 256
        # -------------------------------------------#
        out = self.body.forward(inputs)

        # -------------------------------------------#
        #   获得三个shape的有效特征层
        #   分别是output1  80, 80, 64
        #         output2  40, 40, 64
        #         output3  20, 20, 64
        # -------------------------------------------#
        fpn = self.fpn.forward(out)

        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        # -------------------------------------------#
        #   将所有结果进行堆叠
        # -------------------------------------------#
        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        if self.mode == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
        return output



#FaceUtils/anchors.py
class face_Anchors(object):
    def __init__(self, cfg, image_size=None, phase='train'):
        super(face_Anchors, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        #---------------------------#
        #   图片的尺寸
        #---------------------------#
        self.image_size = image_size
        #---------------------------#
        #   三个有效特征层高和宽
        #---------------------------#
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]

    def get_anchors(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            #-----------------------------------------#
            #   对特征层的高和宽进行循环迭代
            #-----------------------------------------#
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
        
#FaceUtils/box_utils.py
def face_letterbox_image(image, size):
    ih, iw, _ = np.shape(image)
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    new_image = np.ones([size[1], size[0], 3]) * 128
    new_image[(h - nh) // 2:nh + (h - nh) // 2, (w - nw) // 2:nw + (w - nw) // 2] = image
    return new_image

def face_retinaface_correct_boxes(result, input_shape, image_shape):
    new_shape = image_shape * np.min(input_shape / image_shape)

    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape

    scale_for_boxs = [scale[1], scale[0], scale[1], scale[0]]
    scale_for_landmarks = [scale[1], scale[0], scale[1], scale[0], scale[1], scale[0], scale[1], scale[0], scale[1],
                           scale[0]]

    offset_for_boxs = [offset[1], offset[0], offset[1], offset[0]]
    offset_for_landmarks = [offset[1], offset[0], offset[1], offset[0], offset[1], offset[0], offset[1], offset[0],
                            offset[1], offset[0]]

    result[:, :4] = (result[:, :4] - np.array(offset_for_boxs)) * np.array(scale_for_boxs)
    result[:, 5:] = (result[:, 5:] - np.array(offset_for_landmarks)) * np.array(scale_for_landmarks)

    return result

def face_point_form(boxes):
    # ------------------------------#
    #   获得框的左上角和右下角
    # ------------------------------#
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,
                      boxes[:, :2] + boxes[:, 2:] / 2), 1)

def face_center_size(boxes):
    # ------------------------------#
    #   获得框的中心和宽高
    # ------------------------------#
    return torch.cat((boxes[:, 2:] + boxes[:, :2]) / 2,
                     boxes[:, 2:] - boxes[:, :2], 1)

def face_intersect(box_a, box_b):
    # 计算所有真实框和先验框的交面积
    A = box_a.size(0)
    B = box_b.size(0)
    # ------------------------------#
    #   获得交矩形的左上角
    # ------------------------------#
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    # ------------------------------#
    #   获得交矩形的右下角
    # ------------------------------#
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    # -------------------------------------#
    #   计算先验框和所有真实框的重合面积
    # -------------------------------------#
    return inter[:, :, 0] * inter[:, :, 1]

def face_jaccard(box_a, box_b):
    # -------------------------------------#
    #   返回的inter的shape为[A,B]
    #   代表每一个真实框和先验框的交矩形
    # -------------------------------------#
    inter = face_intersect(box_a, box_b)
    # -------------------------------------#
    #   计算先验框和真实框各自的面积
    # -------------------------------------#
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]

    union = area_a + area_b - inter
    # -------------------------------------#
    #   每一个真实框和先验框的交并比[A,B]
    # -------------------------------------#
    return inter / union  # [A,B]

def face_match(threshold, truths, priors, variances, labels, landms, loc_t, conf_t, landm_t, idx):
    # ----------------------------------------------#
    #   计算所有的先验框和真实框的重合程度
    # ----------------------------------------------#
    overlaps = face_jaccard(
        truths,
        face_point_form(priors)
    )
    # ----------------------------------------------#
    #   所有真实框和先验框的最好重合程度
    #   best_prior_overlap [truth_box,1]
    #   best_prior_idx [truth_box,1]
    # ----------------------------------------------#
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)

    # ----------------------------------------------#
    #   所有先验框和真实框的最好重合程度
    #   best_truth_overlap [1,prior]
    #   best_truth_idx [1,prior]
    # ----------------------------------------------#
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)

    # ----------------------------------------------#
    #   用于保证每个真实框都至少有对应的一个先验框
    # ----------------------------------------------#
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)
    # 对best_truth_idx内容进行设置
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j

    # ----------------------------------------------#
    #   获取每一个先验框对应的真实框[num_priors,4]
    # ----------------------------------------------#
    matches = truths[best_truth_idx]
    # Shape: [num_priors] 此处为每一个anchor对应的label取出来
    conf = labels[best_truth_idx]
    matches_landm = landms[best_truth_idx]

    # ----------------------------------------------#
    #   如果重合程度小于threhold则认为是背景
    # ----------------------------------------------#
    conf[best_truth_overlap < threshold] = 0
    # ----------------------------------------------#
    #   利用真实框和先验框进行编码
    #   编码后的结果就是网络应该有的预测结果
    # ----------------------------------------------#
    loc = face_encode(matches, priors, variances)
    landm = face_encode_landm(matches_landm, priors, variances)

    # [num_priors, 4]
    loc_t[idx] = loc
    # [num_priors]
    conf_t[idx] = conf
    # [num_priors, 10]
    landm_t[idx] = landm

def face_encode(matched, priors, variances):
    # 进行编码的操作
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
    # 中心编码
    g_cxcy /= (variances[0] * priors[:, 2:])

    # 宽高编码
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]

def face_encode_landm(matched, priors, variances):
    matched = torch.reshape(matched, (matched.size(0), 5, 2))
    priors_cx = priors[:, 0].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    priors_cy = priors[:, 1].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    priors_w = priors[:, 2].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    priors_h = priors[:, 3].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    priors = torch.cat([priors_cx, priors_cy, priors_w, priors_h], dim=2)

    # 减去中心后除上宽高
    g_cxcy = matched[:, :, :2] - priors[:, :, :2]
    g_cxcy /= (variances[0] * priors[:, :, 2:])
    g_cxcy = g_cxcy.reshape(g_cxcy.size(0), -1)
    return g_cxcy

def face_log_sum_exp(x):
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x - x_max), 1, keepdim=True)) + x_max

def face_decode(loc, priors, variances):
    # 中心解码，宽高解码
    boxes = torch.cat((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
                       priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def face_decode_landm(pre, priors, variances):
    # 关键点解码
    landms = torch.cat((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                        ), dim=1)
    return landms

def face_non_max_suppression(boxes, conf_thres=0.5, nms_thres=0.3):
    detection = boxes
    # 1、找出该图片中得分大于门限函数的框。在进行重合框筛选前就进行得分的筛选可以大幅度减少框的数量。
    mask = detection[:, 4] >= conf_thres
    detection = detection[mask]
    if not np.shape(detection)[0]:
        return []

    best_box = []
    scores = detection[:, 4]
    # 2、根据得分对框进行从大到小排序。
    arg_sort = np.argsort(scores)[::-1]
    detection = detection[arg_sort]

    while np.shape(detection)[0] > 0:
        # 3、每次取出得分最大的框，计算其与其它所有预测框的重合程度，重合程度过大的则剔除。
        best_box.append(detection[0])
        if len(detection) == 1:
            break
        ious = face_iou(best_box[-1], detection[1:])
        detection = detection[1:][ious < nms_thres]

    return np.array(best_box)

def face_iou(b1, b2):
    b1_x1, b1_y1, b1_x2, b1_y2 = b1[0], b1[1], b1[2], b1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = b2[:, 0], b2[:, 1], b2[:, 2], b2[:, 3]

    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)

    inter_area = np.maximum(inter_rect_x2 - inter_rect_x1, 0) * \
                 np.maximum(inter_rect_y2 - inter_rect_y1, 0)

    area_b1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area_b2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / np.maximum((area_b1 + area_b2 - inter_area), 1e-6)
    return iou
        
#FaceUtils/config.py
face_cfg_mnet = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    #------------------------------------------------------------------#
    #   视频上看到的训练图片大小为640，为了提高大图状态下的困难样本
    #   的识别能力，我将训练图片进行调大
    #------------------------------------------------------------------#
    'train_image_size': 960,
    'predict_image_size': 2150,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}

face_cfg_re50 = {
    'name': 'Resnet50',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    #------------------------------------------------------------------#
    #   视频上看到的训练图片大小为840，为了提高大图状态下的困难样本
    #   的识别能力，我将训练图片进行调大
    #------------------------------------------------------------------#
    'train_image_size': 960,
    'predict_image_size': 2150,
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 256,
    'out_channel': 256
}

#FaceRetinaFace.py
def face_preprocess_input(image):
    image -= np.array((104, 117, 123), np.float32)
    return image


# ------------------------------------#
#   请注意主干网络与预训练权重的对应
#   即注意修改model_path和backbone
# ------------------------------------#
class face_Retinaface(object):
    _defaults = {
        "model_path": 'C:/SSM/ObjectsDetection/models/Retinaface_mobilenet0.25.pth',
        "backbone": 'mobilenet',
        "confidence": 0.5,
        "nms_iou": 0.45,
        "cuda": True,
        # ----------------------------------------------------------------------#
        #   是否需要进行图像大小限制。
        #   开启后，会将输入图像的大小限制为input_shape。否则使用原图进行预测。
        #   可根据输入图像的大小自行调整input_shape，注意为32的倍数，如[640, 640, 3]
        # ----------------------------------------------------------------------#
        "input_shape": [1280, 1280, 3],
        "letterbox_image": True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化Retinaface
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        if self.backbone == "mobilenet":
            self.cfg = face_cfg_mnet
        else:
            self.cfg = face_cfg_re50
        self.generate()
        if self.letterbox_image:
            self.anchors = face_Anchors(self.cfg, image_size=[self.input_shape[0], self.input_shape[1]]).get_anchors()

    # ---------------------------------------------------#
    #   载入模型
    # ---------------------------------------------------#
    def generate(self):
        self.net = faceretina_RetinaFace(cfg=self.cfg, mode='eval').eval()

        # -------------------------------#
        #   载入模型与权值
        # -------------------------------#
        # print('Loading weights into state dict...')
        state_dict = torch.load(self.model_path, map_location=torch.device('cpu'))
        self.net.load_state_dict(state_dict)
        if self.cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            self.net = nn.DataParallel(self.net)
            # self.net = self.net.cuda()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.net = self.net.to(device)
        # print('Finished!')

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image):
        # ---------------------------------------------------#
        #   对输入图像进行一个备份，后面用于绘图
        # ---------------------------------------------------#
        old_image = image.copy()

        image = np.array(image, np.float32)

        # ---------------------------------------------------#
        #   计算scale，用于将获得的预测框转换成原图的高宽
        # ---------------------------------------------------#
        scale = [np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]]
        scale_for_landmarks = [np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
                               np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
                               np.shape(image)[1], np.shape(image)[0]]

        im_height, im_width, _ = np.shape(image)
        # ---------------------------------------------------------#
        #   letterbox_image可以给图像增加灰条，实现不失真的resize
        # ---------------------------------------------------------#
        if self.letterbox_image:
            image = np.array(face_letterbox_image(image, [self.input_shape[1], self.input_shape[0]]), np.float32)
        else:
            self.anchors = face_Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()

        with torch.no_grad():
            # -----------------------------------------------------------#
            #   图片预处理，归一化。
            # -----------------------------------------------------------#
            image = torch.from_numpy(face_preprocess_input(image).transpose(2, 0, 1)).unsqueeze(0)

            if self.cuda:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.anchors = self.anchors.to(device)
                image = image.to(device)

            loc, conf, landms = self.net(image)

            # -----------------------------------------------------------#
            #   将预测结果进行解码
            # -----------------------------------------------------------#
            boxes = face_decode(loc.data.squeeze(0), self.anchors, self.cfg['variance'])
            boxes = boxes.cpu().numpy()

            conf = conf.data.squeeze(0)[:, 1:2].cpu().numpy()

            landms = face_decode_landm(landms.data.squeeze(0), self.anchors, self.cfg['variance'])
            landms = landms.cpu().numpy()

            boxes_conf_landms = np.concatenate([boxes, conf, landms], -1)
            boxes_conf_landms = face_non_max_suppression(boxes_conf_landms, self.confidence)
            if len(boxes_conf_landms) <= 0:
                return old_image
            # ---------------------------------------------------------#
            #   如果使用了letterbox_image的话，要把灰条的部分去除掉。
            # ---------------------------------------------------------#
            if self.letterbox_image:
                boxes_conf_landms = face_retinaface_correct_boxes(boxes_conf_landms, \
                                                             np.array([self.input_shape[0], self.input_shape[1]]),
                                                             np.array([im_height, im_width]))

        boxes_conf_landms[:, :4] = boxes_conf_landms[:, :4] * scale
        boxes_conf_landms[:, 5:] = boxes_conf_landms[:, 5:] * scale_for_landmarks
        for b in boxes_conf_landms:
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))

            # b[0]-b[3]为人脸框的坐标，b[4]为得分，b[5]-b[14]为人脸关键点的坐标
            # cv2.rectangle(old_image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            # print("Face:",b[0], b[1], b[2], b[3])
            # print(old_image.shape)
            # 在 10 * 10 的方格内,选取一个像素,进行填充
            if b[2] - b[0] > 10 and b[3] - b[1] > 10:
                for m in range(b[1], b[3], 10):
                    for n in range(b[0], b[2], 10):
                        (rr, gg, bb) = old_image[m, n]
                        for i in range(10):
                            for j in range(10):
                                old_image[i + m, j + n] = (rr, gg, bb)
            else:
                # 选取左下角像素,进行填充
                (rr, gg, bb) = old_image[b[1], b[0]]
                for m in range(b[1], b[3]):
                    for n in range(b[0], b[2]):
                        old_image[m, n] = (rr, gg, bb)
        return old_image


'''*********************************************LP************************************************'''

#LPData/config.py
lp_cfg_mnet = {
    'name': 'mobilenet0.25',
    'min_sizes': [[24, 48], [96, 192], [384, 768]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 48,
    'ngpu': 1,
    'epoch': 50,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': False,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}


#LPLayers/functions/prior_box.py
class lp_PriorBox(object):
    def __init__(self, cfg, image_size=None, phase='train'):
        super(lp_PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        self.name = "s"

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]
                        #print(anchors)
                        #exit(1)

        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output

#LPModels/net.py
def lp_conv_bn(inp, oup, stride = 1, leaky = 0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def lp_conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )

def lp_conv_bn1X1(inp, oup, stride, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def lp_conv_dw(inp, oup, stride, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),
    )

class lp_SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(lp_SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if (out_channel <= 64):
            leaky = 0.1
        self.conv3X3 = lp_conv_bn_no_relu(in_channel, out_channel//2, stride=1)

        self.conv5X5_1 = lp_conv_bn(in_channel, out_channel//4, stride=1, leaky = leaky)
        self.conv5X5_2 = lp_conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

        self.conv7X7_2 = lp_conv_bn(out_channel//4, out_channel//4, stride=1, leaky = leaky)
        self.conv7x7_3 = lp_conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

    def forward(self, input):
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out

class lp_FPN(nn.Module):
    def __init__(self,in_channels_list,out_channels):
        super(lp_FPN,self).__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1
        self.output1 = lp_conv_bn1X1(in_channels_list[0], out_channels, stride = 1, leaky = leaky)
        self.output2 = lp_conv_bn1X1(in_channels_list[1], out_channels, stride = 1, leaky = leaky)
        self.output3 = lp_conv_bn1X1(in_channels_list[2], out_channels, stride = 1, leaky = leaky)

        self.merge1 = lp_conv_bn(out_channels, out_channels, leaky = leaky)
        self.merge2 = lp_conv_bn(out_channels, out_channels, leaky = leaky)

    def forward(self, input):
        # names = list(input.keys())
        input = list(input.values())

        output1 = self.output1(input[0])
        output2 = self.output2(input[1])
        output3 = self.output3(input[2])

        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        out = [output1, output2, output3]
        return out

class lp_MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            lp_conv_bn(3, 8, 2, leaky = 0.1),    # 3
            lp_conv_dw(8, 16, 1),   # 7
            lp_conv_dw(16, 32, 2),  # 11
            lp_conv_dw(32, 32, 1),  # 19
            lp_conv_dw(32, 64, 2),  # 27
            lp_conv_dw(64, 64, 1),  # 43
        )
        self.stage2 = nn.Sequential(
            lp_conv_dw(64, 128, 2),  # 43 + 16 = 59
            lp_conv_dw(128, 128, 1), # 59 + 32 = 91
            lp_conv_dw(128, 128, 1), # 91 + 32 = 123
            lp_conv_dw(128, 128, 1), # 123 + 32 = 155
            lp_conv_dw(128, 128, 1), # 155 + 32 = 187
            lp_conv_dw(128, 128, 1), # 187 + 32 = 219
        )
        self.stage3 = nn.Sequential(
            lp_conv_dw(128, 256, 2), # 219 +3 2 = 241
            lp_conv_dw(256, 256, 1), # 241 + 64 = 301
        )
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256, 1000)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        # x = self.model(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x

#LPModels/retina.py
class lp_ClassHead(nn.Module):

    def __init__(self, inchannels=512, num_anchors=3):
        super(lp_ClassHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels, self.num_anchors * 2, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 2)

class lp_BboxHead(nn.Module):

    def __init__(self, inchannels=512, num_anchors=3):
        super(lp_BboxHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 4, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 4)

class lp_LandmarkHead(nn.Module):

    def __init__(self, inchannels=512, num_anchors=3):
        super(lp_LandmarkHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 8, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 8)

class lp_Retina(nn.Module):

    def __init__(self, cfg=None, phase='train'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(lp_Retina, self).__init__()
        self.phase = phase
        backbone = None
        if cfg['name'] == 'mobilenet0.25':
            backbone = MobileNetV1()

            self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])

        in_channels_stage2 = cfg['in_channel']
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = cfg['out_channel']
        self.fpn = FPN(in_channels_list, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=cfg['out_channel'])

    def _make_class_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(lp_ClassHead(inchannels, anchor_num))
        return classhead

    def _make_bbox_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(lp_BboxHead(inchannels, anchor_num))
        return bboxhead

    def _make_landmark_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(lp_LandmarkHead(inchannels, anchor_num))
        return landmarkhead

    def forward(self, inputs):
        out = self.body(inputs)

        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
        return output

#LPUtils/nms/py_cpu_nms.py
# Fast R-CNN
def lp_py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

# LPUtils/box_utils.py
def lp_point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax

def lp_center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, 2:] + boxes[:, :2])/2,  # cx, cy
                     boxes[:, 2:] - boxes[:, :2], 1)  # w, h

def lp_intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]

def lp_jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = lp_intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]

def lp_matrix_iou(a, b):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i)

def lp_matrix_iof(a, b):
    """
    return iof of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    return area_i / np.maximum(area_a[:, np.newaxis], 1)

def lp_match(threshold, truths, priors, variances, labels, landms, loc_t, conf_t, landm_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, 4].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        landms: (tensor) Ground truth landms, Shape [num_obj, 10].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        landm_t: (tensor) Tensor to be filled w/ endcoded landm targets.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location 2)confidence 3)landm preds.
    """
    # jaccard index
    overlaps = lp_jaccard(
        truths,
        lp_point_form(priors)
    )
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)

    # ignore hard gt
    valid_gt_idx = best_prior_overlap[:, 0] >= 0.2
    best_prior_idx_filter = best_prior_idx[valid_gt_idx, :]
    if best_prior_idx_filter.shape[0] <= 0:
        loc_t[idx] = 0
        conf_t[idx] = 0
        return

    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_idx_filter.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx_filter, 2)  # ensure best prior
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):     # 判别此anchor是预测哪一个boxes
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]            # Shape: [num_priors,4] 此处为每一个anchor对应的bbox取出来
    conf = labels[best_truth_idx]               # Shape: [num_priors]      此处为每一个anchor对应的label取出来
    conf[best_truth_overlap < threshold] = 0    # label as background   overlap<0.35的全部作为负样本
    loc = lp_encode(matches, priors, variances)

    matches_landm = landms[best_truth_idx]
    landm = lp_encode_landm(matches_landm, priors, variances)
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior
    landm_t[idx] = landm

def lp_encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]

def lp_encode_landm(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 10].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded landm (tensor), Shape: [num_priors, 10]
    """
    #print('%%%',matched.shape,matched.size(0))
    # dist b/t match center and prior's center
    matched = torch.reshape(matched, (matched.size(0), 4, 2))
    priors_cx = priors[:, 0].unsqueeze(1).expand(matched.size(0), 4).unsqueeze(2)
    priors_cy = priors[:, 1].unsqueeze(1).expand(matched.size(0), 4).unsqueeze(2)
    priors_w = priors[:, 2].unsqueeze(1).expand(matched.size(0), 4).unsqueeze(2)
    priors_h = priors[:, 3].unsqueeze(1).expand(matched.size(0), 4).unsqueeze(2)
    priors = torch.cat([priors_cx, priors_cy, priors_w, priors_h], dim=2)
    g_cxcy = matched[:, :, :2] - priors[:, :, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, :, 2:])
    # g_cxcy /= priors[:, :, 2:]
    g_cxcy = g_cxcy.reshape(g_cxcy.size(0), -1)
    #print('$$',g_cxcy.shape)
    # return target for smooth_l1_loss
    return g_cxcy

# Adapted from https://github.com/Hakuyume/chainer-ssd
def lp_decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def lp_decode_landm(pre, priors, variances):
    """Decode landm from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        pre (tensor): landm predictions for loc layers,
            Shape: [num_priors,10]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded landm predictions
    """
    landms = torch.cat((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                        #priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                        ), dim=1)
    return landms

def lp_log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max

# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def lp_nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = torch.Tensor(scores.size(0)).fill_(0).long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count


#LPDetect.py



def lpargs():
    lp_parser = argparse.ArgumentParser(description='RetinaPL')
    lp_parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
    lp_parser.add_argument('--top_k', default=1000, type=int, help='top_k')
    lp_parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
    lp_parser.add_argument('--keep_top_k', default=500, type=int, help='keep_top_k')
    lp_parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
    lp_parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
    lp_parser.add_argument('--read_path', nargs='?', type=str,
                             default="C:/SSM/upload/2021571/",
                             help='read_path')
    lp_parser.add_argument('--write_path', nargs='?', type=str,
                             default="C:/SSM/upload/2021571mask/",
                             help='write_path')
    lp_args = lp_parser.parse_args()

    return lp_args


def lp_check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def lp_remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    # print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def lp_load_model(model, pretrained_path, load_to_cpu):
    # print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location="cpu")  # map_location=lambda storage, loc: storage)
    else:
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pretrained_dict = torch.load(pretrained_path,
                                     map_location="cpu")  # map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = lp_remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = lp_remove_prefix(pretrained_dict, 'module.')
    lp_check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def LPRetinaFace(read_path, write_path):
    torch.set_grad_enabled(False)
    cfg = lp_cfg_mnet

    # net and model
    net = lp_Retina(cfg=cfg, phase='test')
    net = lp_load_model(net, 'C:/SSM/ObjectsDetection/models/mnet_plate.pth', False)
    net.eval()
    # print('Finished loading model!')
    cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    resize = 1

    # contents = show_files(read_path, [])#"./img"
    contents = os.listdir(read_path)
    for img_name in contents:
        img_raw = cv2.imread(read_path + img_name, cv2.IMREAD_COLOR)
        img = np.float32(img_raw)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        loc, conf, landms = net(img)  # forward pass

        priorbox = lp_PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = lp_decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()

        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        landms = lp_decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > lpargs().confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:lpargs().top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = lp_py_cpu_nms(dets, lpargs().nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:lpargs().keep_top_k, :]
        landms = landms[:lpargs().keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)
        if lpargs().save_image:
            for b in dets:
                if b[4] < lpargs().vis_thres:
                    continue
                b = list(map(int, b))
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cv2.imwrite(write_path + img_name, img_raw)


def LPRetinaFaceDetect(img_raw):
    torch.set_grad_enabled(False)
    cfg = lp_cfg_mnet
    # net and model
    net = lp_Retina(cfg=cfg, phase='test')
    net = lp_load_model(net, 'C:/SSM/ObjectsDetection/models/mnet_plate.pth', False)
    net.eval()
    # print('Finished loading model!')
    cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    resize = 1
    img = np.float32(img_raw)
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)
    loc, conf, landms = net(img)  # forward pass

    priorbox = lp_PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = lp_decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()

    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

    landms = lp_decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > lpargs().confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:lpargs().top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = lp_py_cpu_nms(dets, lpargs().nms_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:lpargs().keep_top_k, :]
    landms = landms[:lpargs().keep_top_k, :]

    dets = np.concatenate((dets, landms), axis=1)
    if lpargs().save_image:
        for b in dets:
            if b[4] < lpargs().vis_thres:
                continue
            b = list(map(int, b))
            # cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            # 在 10 * 10 的方格内,选取一个像素,进行填充
            if b[2] - b[0] > 10 and b[3] - b[1] > 10:
                for m in range(b[1], b[3], 10):
                    for n in range(b[0], b[2], 10):
                        (bb, gg, rr) = img_raw[m, n]
                        for i in range(10):
                            for j in range(10):
                                img_raw[i + m, j + n] = (bb, gg, rr)
            else:
                # 选取左下角像素,进行填充
                (rr, gg, bb) = img_raw[b[1], b[0]]
                for m in range(b[1], b[3]):
                    for n in range(b[0], b[2]):
                        img_raw[m, n] = (rr, gg, bb)
    return img_raw

'''*********************************************text************************************************'''
#TextModels/fpn_resnet.py
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
          'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3mb4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def text_conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class text_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(text_BasicBlock, self).__init__()
        self.conv1 = text_conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = text_conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class text_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(text_Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=7, scale=1):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels
        self.toplayer_bn = nn.BatchNorm2d(256)
        self.toplayer_relu = nn.ReLU(inplace=True)

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth1_bn = nn.BatchNorm2d(256)
        self.smooth1_relu = nn.ReLU(inplace=True)

        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2_bn = nn.BatchNorm2d(256)
        self.smooth2_relu = nn.ReLU(inplace=True)

        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3_bn = nn.BatchNorm2d(256)
        self.smooth3_relu = nn.ReLU(inplace=True)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer1_bn = nn.BatchNorm2d(256)
        self.latlayer1_relu = nn.ReLU(inplace=True)

        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2_bn = nn.BatchNorm2d(256)
        self.latlayer2_relu = nn.ReLU(inplace=True)

        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3_bn = nn.BatchNorm2d(256)
        self.latlayer3_relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0)

        self.scale = scale

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _upsample(self, x, y, scale=1):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H // scale, W // scale), mode='bilinear', align_corners=True)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        h = x
        h = self.conv1(h)
        h = self.bn1(h)
        h = self.relu1(h)
        h = self.maxpool(h)

        h = self.layer1(h)
        c2 = h
        h = self.layer2(h)
        c3 = h
        h = self.layer3(h)
        c4 = h
        h = self.layer4(h)
        c5 = h

        # Top-down
        p5 = self.toplayer(c5)
        p5 = self.toplayer_relu(self.toplayer_bn(p5))

        c4 = self.latlayer1(c4)
        c4 = self.latlayer1_relu(self.latlayer1_bn(c4))
        p4 = self._upsample_add(p5, c4)
        p4 = self.smooth1(p4)
        p4 = self.smooth1_relu(self.smooth1_bn(p4))

        c3 = self.latlayer2(c3)
        c3 = self.latlayer2_relu(self.latlayer2_bn(c3))
        p3 = self._upsample_add(p4, c3)
        p3 = self.smooth2(p3)
        p3 = self.smooth2_relu(self.smooth2_bn(p3))

        c2 = self.latlayer3(c2)
        c2 = self.latlayer3_relu(self.latlayer3_bn(c2))
        p2 = self._upsample_add(p3, c2)
        p2 = self.smooth3(p2)
        p2 = self.smooth3_relu(self.smooth3_bn(p2))

        p3 = self._upsample(p3, p2)
        p4 = self._upsample(p4, p2)
        p5 = self._upsample(p5, p2)

        out = torch.cat((p2, p3, p4, p5), 1)
        out = self.conv2(out)
        out = self.relu2(self.bn2(out))
        out = self.conv3(out)
        out = self._upsample(out, x, scale=self.scale)

        return out


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(text_BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(text_BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(text_Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_model = model_zoo.load_url(model_urls['resnet50'])
        state = model.state_dict()
        for key in state.keys():
            if key in pretrained_model.keys():
                state[key] = pretrained_model[key]
        model.load_state_dict(state)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(text_Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        pretrained_model = model_zoo.load_url(model_urls['resnet101'])
        state = model.state_dict()
        for key in state.keys():
            if key in pretrained_model.keys():
                state[key] = pretrained_model[key]
        model.load_state_dict(state)
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(text_Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        pretrained_model = model_zoo.load_url(model_urls['resnet152'])
        state = model.state_dict()
        for key in state.keys():
            if key in pretrained_model.keys():
                state[key] = pretrained_model[key]
        model.load_state_dict(state)
    return model



#pse.py
def pse(kernals, min_area):
    kernal_num = len(kernals)
    pred = np.zeros(kernals[0].shape, dtype='int32')

    label_num, label = cv2.connectedComponents(kernals[kernal_num - 1], connectivity=4)

    for label_idx in range(1, label_num):
        if np.sum(label == label_idx) < min_area:
            label[label == label_idx] = 0

    queue = Queue.Queue(maxsize = 0)
    next_queue = Queue.Queue(maxsize = 0)
    points = np.array(np.where(label > 0)).transpose((1, 0))

    for point_idx in range(points.shape[0]):
        x, y = points[point_idx, 0], points[point_idx, 1]
        l = label[x, y]
        queue.put((x, y, l))
        pred[x, y] = l

    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]
    for kernal_idx in range(kernal_num - 2, -1, -1):
        kernal = kernals[kernal_idx].copy()
        while not queue.empty():
            (x, y, l) = queue.get()

            is_edge = True
            for j in range(4):
                tmpx = x + dx[j]
                tmpy = y + dy[j]
                if tmpx < 0 or tmpx >= kernal.shape[0] or tmpy < 0 or tmpy >= kernal.shape[1]:
                    continue
                if kernal[tmpx, tmpy] == 0 or pred[tmpx, tmpy] > 0:
                    continue

                queue.put((tmpx, tmpy, l))
                pred[tmpx, tmpy] = l
                is_edge = False
            if is_edge:
                next_queue.put((x, y, l))

        # kernal[pred > 0] = 0
        queue, next_queue = next_queue, queue

        # points = np.array(np.where(pred > 0)).transpose((1, 0))
        # for point_idx in range(points.shape[0]):
        #     x, y = points[point_idx, 0], points[point_idx, 1]
        #     l = pred[x, y]
        #     queue.put((x, y, l))

    return pred


#mosaic_text_gpu_pypse.py
def scale(img, long_size=1280):
    h, w = img.shape[0:2]
    scale = long_size * 1.0 / max(h, w)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    return img

def text_isPointInRect(x, y, box):
    A = box[0]
    B = box[1]
    C = box[2]
    D = box[3]
    a = int((B[0] - A[0])*(y - A[1]) - (B[1] - A[1])*(x - A[0]))
    b = int((C[0] - B[0])*(y - B[1]) - (C[1] - B[1])*(x - B[0]))
    c = int((D[0] - C[0])*(y - C[1]) - (D[1] - C[1])*(x - C[0]))
    d = int((A[0] - D[0])*(y - D[1]) - (A[1] - D[1])*(x - D[0]))
    if (a>0 and b>0 and c>0 and d>0) or (a<0 and b<0 and c<0 and d<0):
        return True
    return False

def text_mosaic_one_box(img, box):
    rect_minx = int(box[:,0].min())
    rect_miny = int(box[:,1].min())
    rect_maxx = min(int(box[:,0].max()), img.shape[1] - 1)
    rect_maxy = min(int(box[:,1].max()), img.shape[0] - 1)

    MOSAIC_SIZE = 15

    '''if(rect_maxx-rect_minx>MOSAIC_SIZE and rect_maxy-rect_miny>MOSAIC_SIZE):
        for x in range(rect_minx, rect_maxx, MOSAIC_SIZE):
            for y in range(rect_miny, rect_maxy, MOSAIC_SIZE):
                base_color = img[y, x]
                for xx in range(x, min(x + MOSAIC_SIZE, rect_maxx)):
                    for yy in range(y, min(y + MOSAIC_SIZE, rect_maxy)):
                        if not isPointInRect(xx, yy, box):
                            continue
                        img[yy, xx] = base_color
    else:
        base_color = img[rect_miny, rect_minx]
        for x in range(rect_minx, rect_maxx):
            for y in range(rect_miny, rect_maxy):
                        if not isPointInRect(x, y, box):
                            continue
                        img[y, x] = base_color'''
    base_color = img[rect_miny, rect_minx]
    for x in range(rect_minx, rect_maxx):
        for y in range(rect_miny, rect_maxy):
            if not text_isPointInRect(x, y, box):
                continue
            img[y, x] = base_color
    return img

def load_image(img_path, long_size):
    #print(img_path)
    org_img = cv2.imread(img_path)
    # convert BGR to RGB
    img = org_img[:, :, [2, 1, 0]]
    scaled_img = scale(img, long_size)
    scaled_img = Image.fromarray(scaled_img)
    scaled_img = scaled_img.convert('RGB')
    # ndarray -> tensor    H*W*C -> C*H*W
    scaled_img = transforms.ToTensor()(scaled_img)
    scaled_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(scaled_img)
    # C*H*W -> 1*C*H*W
    scaled_img = scaled_img.unsqueeze(0)
    return org_img, scaled_img

def load_model(args):
    model = resnet50(pretrained=True, num_classes=7, scale=args.scale)
    for param in model.parameters():
        param.requires_grad = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=model.to(device)
    #model = model.cuda()
    assert os.path.exists(args.resume), '[ERROR] No checkpoint found at {}'.format(args.resume)
    #print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
    sys.stdout.flush()
    checkpoint = torch.load(args.resume, map_location='cpu')
    d = collections.OrderedDict()
    for key, value in checkpoint['state_dict'].items():
        tmp = key[7:]
        d[tmp] = value
    model.load_state_dict(d)
    #print("Loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    sys.stdout.flush()
    model.eval()
    return model

def detect_text(args, img_path, model):
    org_img, img = load_image(img_path, args.long_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        img = Variable(img.to(device))

    #img = Variable(img.cuda(), volatile=True)
    org_img = org_img.astype('uint8')
    text_box = org_img.copy()

    outputs = model(img)
    score = torch.sigmoid(outputs[:, 0, :, :])
    outputs = (torch.sign(outputs - args.binary_th) + 1) / 2
    text = outputs[:, 0, :, :]
    kernels = outputs[:, 0:args.kernel_num, :, :] * text
    score = score.data.cpu().numpy()[0].astype(np.float32)
    text = text.data.cpu().numpy()[0].astype(np.uint8)
    kernels = kernels.data.cpu().numpy()[0].astype(np.uint8)
    # python version pse
    pred = pse(kernels, args.min_kernel_area / (args.scale * args.scale))
    scale = (org_img.shape[1] * 1.0 / pred.shape[1], org_img.shape[0] * 1.0 / pred.shape[0])
    label = pred
    label_num = np.max(label) + 1
    bboxes = []
    for i in range(1, label_num):
        points = np.array(np.where(label == i)).transpose((1, 0))[:, ::-1]
        if points.shape[0] < args.min_area / (args.scale * args.scale):
            continue
        score_i = np.mean(score[label == i])
        if score_i < args.min_score:
            continue
        rect = cv2.minAreaRect(points)
        bbox = cv2.boxPoints(rect) * scale
        bbox = bbox.astype('int32')
        bboxes.append(bbox.reshape(-1))

    sys.stdout.flush()

    return text_box, bboxes

def text_create_mosaic(image, bboxes, draw_contour=False):
    for bbox in bboxes:
        image = text_mosaic_one_box(image, bbox.reshape(4, 2))
        if draw_contour:
            cv2.drawContours(image, [bbox.reshape(4, 2)], -1, (0, 255, 0), 2)
    #print("text mosaic!")
    return image

def TextDetect():
    text_parser = argparse.ArgumentParser(description='Hyperparams')
    text_parser.add_argument('--arch', nargs='?', type=str, default='resnet50')
    text_parser.add_argument('--binary_th', nargs='?', type=float, default=1.0,
                        help='binary threshold')
    text_parser.add_argument('--kernel_num', nargs='?', type=int, default=7,
                        help='kernel num')
    text_parser.add_argument('--scale', nargs='?', type=int, default=1,
                        help='scale')
    text_parser.add_argument('--min_kernel_area', nargs='?', type=float, default=5.0,
                        help='min kernel area')
    text_parser.add_argument('--min_area', nargs='?', type=float, default=800.0,
                        help='min area')
    text_parser.add_argument('--min_score', nargs='?', type=float, default=0.93,
                        help='min score')
    text_parser.add_argument('--resume', nargs='?', type=str, default="C:/SSM/ObjectsDetection/models/ctw1500_res50_pretrain_ic17.pth.tar",
                        help='Path to previous saved model to restart from')
    text_parser.add_argument('--long_size', nargs='?', type=int, default=1280,
                        help='The longer size of resized image')
    text_parser.add_argument('--read_path', nargs='?', type=str,
                             default="C:/SSM/upload/2021571/",#"C:/Users/prin/Desktop/1/",
                             help='read_path')
    text_parser.add_argument('--write_path', nargs='?', type=str,
                             default="C:/SSM/upload/2021571mask/",#"C:/Users/prin/Desktop/1mask/",
                             help='write_path')
    text_args = text_parser.parse_args()
    text_model = load_model(text_args)
    return text_args,text_model
        
        
#print(torch.__version__)
#print(img.shape)
print("start!!!!!")
retinaface = face_Retinaface()
print("face model loaded")
text_args, text_model = TextDetect()
print("text model loaded")
read_path=text_args.read_path
write_path=text_args.write_path
print(read_path,write_path)
contents = os.listdir(read_path)
img_num=len(contents)
print("共需处理",img_num,"个图片")
img_nn=0
for file in contents:
    img_path = read_path + file
    print("img_path:",img_path)
    # text
    img, bboxes = detect_text(text_args, img_path, text_model)
    img_text = text_create_mosaic(img, bboxes)
    # face
    img_face = cv2.cvtColor(img_text, cv2.COLOR_BGR2RGB)
    img_face = retinaface.detect_image(img_face)
    img_face = cv2.cvtColor(img_face, cv2.COLOR_RGB2BGR)
    # LP
    img_lp = LPRetinaFaceDetect(img_face)
    cv2.imwrite(write_path + file, img_lp)
    img_nn+=1
    print("已处理完",img_nn,"个图片")