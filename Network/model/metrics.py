#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from ignite.exceptions import NotComputableError
from ignite.metrics.accumulation import VariableAccumulation


def intersection(bbox_pred, bbox_gt):
    max_xy = torch.min(bbox_pred[:, 2:], bbox_gt[:, 2:])
    min_xy = torch.max(bbox_pred[:, :2], bbox_gt[:, :2])
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, 0] * inter[:, 1]


def jaccard(bbox_pred, bbox_gt):
    inter = intersection(bbox_pred, bbox_gt)
    area_pred = (bbox_pred[:, 2] - bbox_pred[:, 0]) * (bbox_pred[:, 3] -
                                                       bbox_pred[:, 1])
    area_gt = (bbox_gt[:, 2] - bbox_gt[:, 0]) * (bbox_gt[:, 3] -
                                                 bbox_gt[:, 1])
    union = area_pred + area_gt - inter
    iou = torch.div(inter, union)
    return torch.sum(iou), (iou > 0.5).sum().item(), (iou > 0.3).sum().item()

def iou(bbox_pred, bbox_gt):
    inter = intersection(bbox_pred, bbox_gt)
    area_pred = (bbox_pred[:, 2] - bbox_pred[:, 0]) * (bbox_pred[:, 3] -
        bbox_pred[:, 1])
    area_gt = (bbox_gt[:, 2] - bbox_gt[:, 0]) * (bbox_gt[:, 3] -
        bbox_gt[:, 1])
    union = area_pred + area_gt - inter
    iou = torch.div(inter, union).view(-1,1)
    return iou

class MetricAverage(VariableAccumulation):
    def __init__(self, output_transform=lambda x: x):
        def _mean_op(a, x):
            return a+(x.sum().item())
        super(MetricAverage, self).__init__(op=_mean_op, output_transform=output_transform)

    def compute(self):
        if self.num_examples < 1:
            raise NotComputableError("{} must have at least one example before"
                                    " it can be computed.".format(self.__class__.__name__))

        return self.accumulator / self.num_examples

def image_acc(y_pred,y):
    B,H,W = y.shape
    indices = y_pred
    if y_pred.dim() == y.dim()+1:
        indices = torch.argmax(y_pred.softmax(1), dim=1)
    count = H*W
    correct = torch.eq(indices.float(),y.float()).sum([1,2])
    acc = correct.float()/count
    return acc.view(-1,1)

def image_acc_ignore(y_pred,y,ignore_index):
    B,H,W = y.shape
    indices = y_pred
    if y_pred.dim() == y.dim()+1:
        indices = torch.argmax(y_pred.softmax(1), dim=1)
    masks = y.ne(ignore_index)
    count = masks.sum([1,2])
    correct = torch.zeros(B).to(count)
    for i in range(y.shape[0]):
        y_i = y[i].masked_select(masks[i])
        y_pred_i = indices[i].masked_select(masks[i])
        correct[i]=torch.eq(y_pred_i, y_i).sum()
    acc = correct.float()/count.float()
    return acc.view(-1,1)

def binary_image_acc(y_pred,y):
    B,H,W = y.shape
    count = H*W
    correct = torch.eq(y_pred.float(),y.float()).sum([1,2])
    acc = correct.float()/count 
    return acc.view(-1,1)

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('Accuracy must have at least one example before it can be computed.')
        return self._num_correct / self._num_examples
