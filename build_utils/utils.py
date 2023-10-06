import glob
import math
import os
import random
import time

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm

from build_utils import torch_utils  # , google_utils

# Set printoptions
torch.set_printoptions(linewidth=320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
matplotlib.rc('font', **{'size': 11})

# Prevent OpenCV from multithreading (to use PyTorch DataLoader)
cv2.setNumThreads(0)


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch_utils.init_seeds(seed=seed)


def check_file(file):
    # Searches for file if not found locally
    if os.path.isfile(file):
        return file
    else:
        files = glob.glob('./**/' + file, recursive=True)  # find file
        assert len(files), 'File Not Found: %s' % file  # assert file was found
        return files[0]  # return first file if multiple found


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    将预测的坐标信息转换回原图尺度
    :param img1_shape: 缩放后的图像尺度
    :param coords: 预测的box信息
    :param img0_shape: 缩放前的图像尺度
    :param ratio_pad: 缩放过程中的缩放比例以及pad
    :return:
    """
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = max(img1_shape) / max(img0_shape)  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


# def bbox_iou(box1, box2, x1y1x2y2=True, GIoU = False, DIoU = False, CIoU = False, alpha=1,  SIoU = False,EIoU = False, WIoU = False, scale=False, Focal = False, gamma = 0.5, eps=1e-7):
#     # Returns the IoU of box1 to box2. box1 is 1*4, box2 is nx4
#     box2 = box2.t()
#     # Get the coordinates of bounding boxes
#     if x1y1x2y2:  # x1, y1, x2, y2 = box1
#         # b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
#         # b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
#
#         (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4,-1)  # chunk() 将张量拆分成指定数量的数据块，每个数据块都是输入张量的一个视图
#         w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
#
#         b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
#         b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
#     else:  # transform from xywh to xyxy
#         # b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
#         # b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
#         # b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
#         # b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
#
#         b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
#         b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
#         w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps)  # 将输入input张量每个元素的值压缩到区间 [min,max]，并返回结果到一个新张量
#         w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps)
#
#     # Intersection area
#     inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
#             (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
#
#     # Union Area
#     w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
#     w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
#     union = (w1 * h1 + eps) + w2 * h2 - inter
#
#     if scale:
#         self = WIoU_Scale(1 - (inter / union))
#
#     # IOU
#     # iou = inter / union  # iou
#
#     iou = torch.pow(inter / (union + eps), alpha)  # alpha iou
#     if GIoU or DIoU or CIoU or EIoU or SIoU or WIoU:
#         cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
#         ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
#         if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
#             c_area = cw * ch + eps  # convex area
#             # return iou - (c_area - union) / c_area  # GIoU
#             return iou - torch.pow((c_area - union) / c_area + eps, alpha)  # GIoU https://arxiv.org/pdf/1902.09630.pdf
#         if DIoU or CIoU or EIoU or SIoU or WIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
#             # convex diagonal squared
#             c2 = (cw ** 2 + ch ** 2) ** alpha + eps # centerpoint distance squared
#             rho2 = (((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4) ** alpha
#             if DIoU:
#                 return iou - rho2 / c2  # DIoU
#             elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
#                 v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
#                 with torch.no_grad():
#                     alpha_ciou = v / (1 - iou + v + eps)
#                 if Focal:
#                     return iou - (rho2 / c2 + torch.pow(v * alpha_ciou + eps, alpha)), torch.pow(inter/(union + eps), gamma)  # Focal_CIoU
#                 else:
#                     return iou - (rho2 / c2 + v * alpha)  # CIoU
#             elif EIoU:
#                 rho_w2 = ((b2_x2 - b2_x1) - (b1_x2 - b1_x1)) ** 2
#                 rho_h2 = ((b2_y2 - b2_y1) - (b1_y2 - b1_y1)) ** 2
#                 cw2 = torch.pow(cw ** 2 + eps, alpha)
#                 ch2 = torch.pow(ch ** 2 + eps, alpha)
#                 if Focal:
#                     return iou - (rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2), torch.pow(inter / (union + eps),gamma)  # Focal_EIou
#                 else:
#                     return iou - (rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2)  # EIou
#             elif SIoU:
#                 # SIoU Loss https://arxiv.org/pdf/2205.12740.pdf
#                 s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5 + eps
#                 s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5 + eps
#                 sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
#                 sin_alpha_1 = torch.abs(s_cw) / sigma
#                 sin_alpha_2 = torch.abs(s_ch) / sigma
#                 threshold = pow(2, 0.5) / 2
#                 sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
#                 angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
#                 rho_x = (s_cw / cw) ** 2
#                 rho_y = (s_ch / ch) ** 2
#                 gamma = angle_cost - 2
#                 distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
#                 omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
#                 omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
#                 shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
#                 if Focal:
#                     return iou - torch.pow(0.5 * (distance_cost + shape_cost) + eps, alpha), torch.pow(inter/(union + eps), gamma) # Focal_SIou
#                 else:
#                     return iou - torch.pow(0.5 * (distance_cost + shape_cost) + eps, alpha) # SIou
#             elif WIoU:
#                 if Focal:
#                     raise RuntimeError("WIoU do not support Focal.")
#                 elif scale:
#                     return getattr(WIoU_Scale, '_scaled_loss')(self), (1 - iou) * torch.exp((rho2 / c2)), iou # WIoU https://arxiv.org/abs/2301.10051
#                 else:
#                     return iou, torch.exp((rho2 / c2))  # WIoU v1
#             if Focal:
#                 return iou - rho2 / c2, torch.pow(inter / (union + eps), gamma)  # Focal_DIoU
#
#         c_area = cw * ch + eps  # convex area
#         if Focal:
#             return iou - torch.pow((c_area - union) / c_area + eps, alpha), torch.pow(inter/(union + eps), gamma)  # Focal_GIoU https://arxiv.org/pdf/1902.09630.pdf
#         else:  # GIoU
#             return iou - torch.pow((c_area - union) / c_area + eps, alpha)  # GIoU https://arxiv.org/pdf/1902.09630.pdf
#     if Focal:
#         return iou, torch.pow(inter / (union + eps), gamma)  # Focal_IoU
#     else:
#         return iou
#


class WIoU_Scale:
    ''' monotonous: {
            None: origin None
            True: monotonic FM v2
            False: non-monotonic FM v3
        }
        momentum: The momentum of running mean'''

    iou_mean = 1.  # 即LIoU 的滑动平均值，每次程序刚开始运行时初始化为 1。如果训练中断导致该值重置，需要将该值恢复为中断前的值，否则会导致性能增速下降
    monotonous = False
    _momentum = 1 - 0.5 ** (1 / 7000)  # 当m足够小时，验证集的IoU基本不影响LIoU的值，此时不需要使用eval和train函数指定训练模式，否则需要使用eval和train函数指定训练模式
    _is_train = True

    def __init__ (self, iou):
        self.iou = iou
        self._update(self)

    @classmethod
    def _update (cls, self):
        if cls._is_train: cls.iou_mean = (1 - cls._momentum) * cls.iou_mean + \
                                         cls._momentum * self.iou.detach().mean().item()

    @classmethod
    def _scaled_loss (cls, self, gamma=1.9, delta=3):  # 聚焦机制对边界框损失的值进行缩放
        if isinstance(self.monotonous, bool):
            if self.monotonous:
                return (self.iou.detach() / self.iou_mean).sqrt()
            else:
                beta = self.iou.detach() / self.iou_mean
                alpha = delta * torch.pow(gamma, beta - delta)
                return beta / alpha
        return 1



def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, SIoU=False, EIoU=False, WIoU=False, Focal=False, alpha=1, gamma=0.5, scale=False, eps=1e-7):
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)
    box2 = box2.t()

    # Get the coordinates of bounding boxes  获取边界框的坐标
    if xywh:  # transform from xywh to xyxy
        # chunk()将张量拆分成指定数量的数据块 每个数据块都是输入张量的一个视图
        b1_x1, b1_x2, b1_y1, b1_y2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2, box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2, b2_y1, b2_y2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2, box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)  # b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)  # b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps)  # 将输入input张量每个元素的值压缩到区间 [min,max]，并返回结果到一个新张量
    w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps)
    union = w1 * h1 + w2 * h2 - inter + eps

    if scale:
        self = WIoU_Scale(1 - (inter / union))

    # IoU
    # iou = inter / union # ori iou
    iou = torch.pow(inter/(union + eps), alpha) # alpha iou
    if CIoU or DIoU or GIoU or EIoU or SIoU or WIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU or EIoU or SIoU or WIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = (cw ** 2 + ch ** 2) ** alpha + eps  # convex diagonal squared
            rho2 = (((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4) ** alpha  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha_ciou = v / (v - iou + (1 + eps))
                if Focal:
                    return iou - (rho2 / c2 + torch.pow(v * alpha_ciou + eps, alpha)), torch.pow(inter/(union + eps), gamma)  # Focal_CIoU
                else:
                    return iou - (rho2 / c2 + torch.pow(v * alpha_ciou + eps, alpha))  # CIoU
            elif EIoU:
                rho_w2 = ((b2_x2 - b2_x1) - (b1_x2 - b1_x1)) ** 2
                rho_h2 = ((b2_y2 - b2_y1) - (b1_y2 - b1_y1)) ** 2
                cw2 = torch.pow(cw ** 2 + eps, alpha)
                ch2 = torch.pow(ch ** 2 + eps, alpha)
                if Focal:
                    return iou - (rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2), torch.pow(inter/(union + eps), gamma) # Focal_EIou
                else:
                    return iou - (rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2) # EIou
            elif SIoU:
                # SIoU Loss https://arxiv.org/pdf/2205.12740.pdf
                s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5 + eps
                s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5 + eps
                sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
                sin_alpha_1 = torch.abs(s_cw) / sigma
                sin_alpha_2 = torch.abs(s_ch) / sigma
                threshold = pow(2, 0.5) / 2
                sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
                angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
                rho_x = (s_cw / cw) ** 2
                rho_y = (s_ch / ch) ** 2
                gamma = angle_cost - 2
                distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
                omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
                omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
                shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
                if Focal:
                    return iou - torch.pow(0.5 * (distance_cost + shape_cost) + eps, alpha), torch.pow(inter/(union + eps), gamma) # Focal_SIou
                else:
                    return iou - torch.pow(0.5 * (distance_cost + shape_cost) + eps, alpha) # SIou
            elif WIoU:
                # print("WIOU！")
                if Focal:
                    raise RuntimeError("WIoU do not support Focal.")
                elif scale:
                    return getattr(WIoU_Scale, '_scaled_loss')(self), (1 - iou) * torch.exp((rho2 / c2)), iou # WIoU https://arxiv.org/abs/2301.10051
                else:
                    return iou, torch.exp((rho2 / c2)) # WIoU v1
            if Focal:
                return iou - rho2 / c2, torch.pow(inter/(union + eps), gamma)  # Focal_DIoU
            else:
                return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        if Focal:
            return iou - torch.pow((c_area - union) / c_area + eps, alpha), torch.pow(inter/(union + eps), gamma)  # Focal_GIoU https://arxiv.org/pdf/1902.09630.pdf
        else:  # GIoU
            return iou - torch.pow((c_area - union) / c_area + eps, alpha)  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    if Focal:
        return iou, torch.pow(inter/(union + eps), gamma)  # Focal_IoU  原论文中glamma = 2.0  a = 0.25时效果更好
    else:
        return iou  # IoU




def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.t())
    area2 = box_area(box2.t())

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def wh_iou(wh1, wh2):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


def compute_loss(p, targets, model):  # predictions, targets, model
    device = p[0].device
    lcls = torch.zeros(1, device=device)  # Tensor(0)       分类损失
    lbox = torch.zeros(1, device=device)  # Tensor(0)       定位损失
    lobj = torch.zeros(1, device=device)  # Tensor(0)       object损失 初始化全部为0
    tcls, tbox, indices, anchors = build_targets(p, targets, model)  # targets
    h = model.hyp  # hyperparameters
    red = 'mean'  # Loss reduction (sum or mean)

    # Define criteria  一个分类损失 一个object损失
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device), reduction=red)
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device), reduction=red)

    # class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
    cp, cn = smooth_BCE(eps=0.0)

    # focal loss
    g = h['fl_gamma']  # focal loss gamma   如果传入的fl_gamma参数是大于0的那么是会使用focal loss损失
    if g > 0:
        BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

    # per output
    for i, pi in enumerate(p):  # layer index, layer predictions
        b, a, gj, gi = indices[i]  # image_idx, anchor_idx, grid_y, grid_x
        # tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
        tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=device)  # target obj

        nb = b.shape[0]  # number of positive samples
        if nb:
            # 对应匹配到正样本的预测信息
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
            pxy = ps[:, :2].sigmoid() * 2 - 0.5  # x 方向上的调整 0~1  -->  -0.5~1.5 只负责这一定范围内目标的预测
            pwh = ps[:, 2:4].exp().clamp(max=1E3) * anchors[i] #
            # ps[:, 2:4].sigmoid() * 2) ** 2的范围是0~4， 再乘以anchors[i]，
            # 表示把预测框的宽和高限制在4倍的anchors内，这是为了解决yolov3和yolov4对预测框宽高无任何约束的问题，这个4和默认的超参数anchor_t是相等的，先验框调整的范围在0-4之间
            # pwh = ps[:, 2:4].exp().clamp(max=1E3) * anchors[i]
            # pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions
            '''
            pxy加sigmoid是为了让值变为0-1之间数值，pxy是小数，就是相对于某个单元格是小数坐标。
            单元格是相应位置，已经根据gj，gi获取到了，ps = pi[b, a, gj, gi]
            就是代表着坐标【gi,gj】，你这个位置来负责和目标gt一样！

            pwh同样需要sigmoid把值归一化到0-1之间，然后乘上anchors[i]，因为anchor的长宽与gt相差不大了，就是4倍左右。
            所以把网络预测值×2再平方  
            [0-1] --> [0,2] -->[0,4]  ||||  (ps[:, 2:4].sigmoid() * 2) ** 2 
            [0-1] --> [0,3]
            '''
            # iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)

            # GIoU
            # pxy = ps[:, :2].sigmoid()
            # pwh = ps[:, 2:4].exp().clamp(max=1E3) * anchors[i]
            pbox = torch.cat((pxy, pwh), 1)  # predicted box
            # giou = bbox_iou(pbox.t(), tbox[i], x1y1x2y2=False, GIoU=True)  # giou(prediction, target)
            # lbox += (1.0 - giou).mean()  # giou loss

            # iou = bbox_iou(pbox.t(), tbox[i], SIoU=True)  # giou(prediction, target)
            iou = bbox_iou(pbox.t(), tbox[i], GIoU=True,scale=True)  # giou(prediction, target)  ,WIoU=True,scale=True
            if type(iou) is tuple:
                if len(iou) == 2:
                    lbox += (iou[1].detach().squeeze() * (1 - iou[0].squeeze())).mean()  # squeeze()降维操作
                    iou = iou[0].squeeze()
                else:
                    lbox += (iou[0] * iou[1]).mean()
                    iou = iou[2].squeeze()
            else:
                lbox += (1.0 - iou.squeeze()).mean()  # iou loss
                iou = iou.squeeze()

            # Obj
            iou = iou.detach().clamp(0).type(tobj.dtype)
            if model.gr < 1:
                iou = (1.0 - model.gr) + model.gr * iou
            tobj[b, a, gj, gi] = iou  # iou ratio


            tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * iou.detach().clamp(0).type(tobj.dtype)  # giou ratio

            # Class
            if model.nc > 1:  # cls loss (only if multiple classes)
                t = torch.full_like(ps[:, 5:], cn, device=device)  # targets
                t[range(nb), tcls[i]] = cp  # class positive
                lcls += BCEcls(ps[:, 5:], t)  # BCE  当检测的目标只有一类时是不会去计算的

            # Append targets to text file
            # with open('targets.txt', 'a') as file:
            #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

        lobj += BCEobj(pi[..., 4], tobj)  # obj loss


    # 乘上每种损失的对应权重
    lbox *= h['box']
    lobj *= h['obj']
    lcls *= h['cls']
    # loss = lbox + lobj + lcls
    return {"box_loss": lbox,
            "obj_loss": lobj,
            "class_loss": lcls}


def build_targets(p, targets, model): # 匹配正样本
    # Build targets for compute_loss(), input targets(image_idx,class,x,y,w,h)
    nt = targets.shape[0]
    tcls, tbox, indices, anch = [], [], [], []
    #gain = torch.ones(6, device=targets.device)  # normalized to gridspace gain
    gain = torch.ones(6, device=targets.device).long()  # normalized to gridspace gain

    multi_gpu = type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
    for i, j in enumerate(model.yolo_layers):  # j: [89, 101, 113]
        # 获取该yolo predictor对应的anchors
        # 注意anchor_vec是anchors缩放到对应特征层上的尺度
        anchors= model.module.module_list[j].anchor_vec if multi_gpu else model.module_list[j].anchor_vec
        # p[i].shape: [batch_size, 3, grid_h, grid_w, num_params]
        gain[2:] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
        na = anchors.shape[0]  # number of anchors
        # [3] -> [3, 1] -> [3, nt]
        at = torch.arange(na).view(na, 1).repeat(1, nt)  # anchor tensor, same as .repeat_interleave(nt)

        # Match targets to anchors
        a, t, offsets = [], targets * gain, 0
        if nt:  # 如果存在target的话
            # 通过计算anchor模板与所有target的wh_iou来匹配正样本
            # j: [3, nt] , iou_t = 0.20
            j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n) = wh_iou(anchors(3,2), gwh(n,2))
            # t.repeat(na, 1, 1): [nt, 6] -> [3, nt, 6]
            # 获取正样本对应的anchor模板与target信息
            at = at.to(j.device)
            a, t = at[j], t.repeat(na, 1, 1)[j]  # filter

        # Define
        # long等于to(torch.int64), 数值向下取整
        b, c = t[:, :2].long().T  # image_idx, class
        gxy = t[:, 2:4]  # grid xy
        gwh = t[:, 4:6]  # grid wh
        gij = (gxy - offsets).long()  # 匹配targets所在的grid cell左上角坐标
        gi, gj = gij.T  # grid xy indices

        # Append
        # gain[3]: grid_h, gain[2]: grid_w
        # image_idx, anchor_idx, grid indices(y, x)
        indices.append((b, a, gj.clamp_(0, gain[3]-1), gi.clamp_(0, gain[2]-1)))
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # gt box相对anchor的x,y偏移量以及w,h
        anch.append(anchors[a])  # anchors
        tcls.append(c)  # class
        if c.shape[0]:  # if any targets
            # 目标的标签数值不能大于给定的目标类别数
            assert c.max() < model.nc, 'Model accepts %g classes labeled from 0-%g, however you labelled a class %g. ' \
                                       'See https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data' % (
                                           model.nc, model.nc - 1, c.max())

    return tcls, tbox, indices, anch


def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6,
                        multi_label=True, classes=None, agnostic=False, max_num=50):
    """
    非极大值抑制部分        conf_thres:置信度和类别阈值     Iou_thres:iou阈值    max_num:NMS后最多只保留前max_num个目标信息 100
    Performs  Non-Maximum Suppression on inference results

    param: prediction[batch, num_anchors, (num_classes+1+4) x num_anchors]
    Returns detections with shape:
        nx6 (x1, y1, x2, y2, conf, cls)
    """

    # Settings
    merge = False  # merge for best mAP
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    time_limit = 10.0  # seconds to quit after

    t = time.time()
    nc = prediction[0].shape[1] - 5  # number of classes
    multi_label &= nc > 1  # multiple labels per box
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference 遍历每张图片
        # Apply constraints
        x = x[x[:, 4] > conf_thres]  # confidence 根据obj confidence虑除背景目标
        x = x[((x[:, 2:4] > min_wh) & (x[:, 2:4] < max_wh)).all(1)]  # width-height 虑除小目标

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[..., 5:] *= x[..., 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:  # 针对每个类别执行非极大值抑制
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).t()
            x = torch.cat((box[i], x[i, j + 5].unsqueeze(1), j.float().unsqueeze(1)), 1)
        else:  # best class only  直接针对每个类别中概率最大的类别进行非极大值抑制处理
            conf, j = x[:, 5:].max(1)
            x = torch.cat((box, conf.unsqueeze(1), j.float().unsqueeze(1)), 1)[conf > conf_thres]

        # Filter by class
        if classes:
            x = x[(j.view(-1, 1) == torch.tensor(classes, device=j.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5] * 0 if agnostic else x[:, 5]  # classes
        boxes, scores = x[:, :4].clone() + c.view(-1, 1) * max_wh, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        i = i[:max_num]  # 最多只保留前max_num个目标信息
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                # i = i[iou.sum(1) > 1]  # require redundancy
            except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(x, i, x.shape, i.shape)
                pass

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


def get_yolo_layers(model):
    bool_vec = [x['type'] == 'yolo' for x in model.module_defs]
    return [i for i, x in enumerate(bool_vec) if x]  # [82, 94, 106] for yolov3


def kmean_anchors(path='./data/coco64.txt', n=9, img_size=(608, 608), thr=0.20, gen=1000):
    # Creates kmeans anchors for use in *.cfg files: from build_utils.build_utils import *; _ = kmean_anchors()
    # n: number of anchors
    # img_size: (min, max) image size used for multi-scale training (can be same values)
    # thr: IoU threshold hyperparameter used for training (0.0 - 1.0)
    # gen: generations to evolve anchors using genetic algorithm
    from build_utils.datasets import LoadImagesAndLabels

    def print_results(k):
        k = k[np.argsort(k.prod(1))]  # sort small to large
        iou = wh_iou(wh, torch.Tensor(k))
        max_iou = iou.max(1)[0]
        bpr, aat = (max_iou > thr).float().mean(), (iou > thr).float().mean() * n  # best possible recall, anch > thr
        print('%.2f iou_thr: %.3f best possible recall, %.2f anchors > thr' % (thr, bpr, aat))
        print('n=%g, img_size=%s, IoU_all=%.3f/%.3f-mean/best, IoU>thr=%.3f-mean: ' %
              (n, img_size, iou.mean(), max_iou.mean(), iou[iou > thr].mean()), end='')
        for i, x in enumerate(k):
            print('%i,%i' % (round(x[0]), round(x[1])), end=',  ' if i < len(k) - 1 else '\n')  # use in *.cfg
        return k

    def fitness(k):  # mutation fitness
        iou = wh_iou(wh, torch.Tensor(k))  # iou
        max_iou = iou.max(1)[0]
        return (max_iou * (max_iou > thr).float()).mean()  # product

    # Get label wh
    wh = []
    dataset = LoadImagesAndLabels(path, augment=True, rect=True)
    nr = 1 if img_size[0] == img_size[1] else 10  # number augmentation repetitions
    for s, l in zip(dataset.shapes, dataset.labels):
        wh.append(l[:, 3:5] * (s / s.max()))  # image normalized to letterbox normalized wh
    wh = np.concatenate(wh, 0).repeat(nr, axis=0)  # augment 10x
    wh *= np.random.uniform(img_size[0], img_size[1], size=(wh.shape[0], 1))  # normalized to pixels (multi-scale)
    wh = wh[(wh > 2.0).all(1)]  # remove below threshold boxes (< 2 pixels wh)

    # Kmeans calculation
    from scipy.cluster.vq import kmeans
    print('Running kmeans for %g anchors on %g points...' % (n, len(wh)))
    s = wh.std(0)  # sigmas for whitening
    k, dist = kmeans(wh / s, n, iter=30)  # points, mean distance
    k *= s
    wh = torch.Tensor(wh)
    k = print_results(k)

    # # Plot
    # k, d = [None] * 20, [None] * 20
    # for i in tqdm(range(1, 21)):
    #     k[i-1], d[i-1] = kmeans(wh / s, i)  # points, mean distance
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    # ax = ax.ravel()
    # ax[0].plot(np.arange(1, 21), np.array(d) ** 2, marker='.')
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7))  # plot wh
    # ax[0].hist(wh[wh[:, 0]<100, 0],400)
    # ax[1].hist(wh[wh[:, 1]<100, 1],400)
    # fig.tight_layout()
    # fig.savefig('wh.png', dpi=200)

    # Evolve
    npr = np.random
    f, sh, mp, s = fitness(k), k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
    for _ in tqdm(range(gen), desc='Evolving anchors'):
        v = np.ones(sh)
        while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
            v = ((npr.random(sh) < mp) * npr.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg = fitness(kg)
        if fg > f:
            f, k = fg, kg.copy()
            print_results(k)
    k = print_results(k)

    return k
