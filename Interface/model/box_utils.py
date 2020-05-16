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

"""
Utilities for dealing with bounding boxes
"""


def box_abs2rel(boxes, inside_boxes, obj_to_img):
  inside_boxes = inside_boxes[obj_to_img]
  ix0, iy0, ix1, iy1 = inside_boxes[:, 0], inside_boxes[:, 1], inside_boxes[:, 2], inside_boxes[:, 3]
  xc = (boxes[:, 0] - ix0) / (ix1 - ix0)
  yc = (boxes[:, 1] - iy0) / (iy1 - iy0)
  w = boxes[:, 2] / (ix1 - ix0)
  h = boxes[:, 3] / (iy1 - iy0)
  return torch.stack([xc, yc, w, h], dim=1)


def box_rel2abs(boxes, inside_boxes, obj_to_img):
  inside_boxes = inside_boxes[obj_to_img]
  ix0, iy0, ix1, iy1 = inside_boxes[:, 0], inside_boxes[:, 1], inside_boxes[:, 2], inside_boxes[:, 3]
  xc = boxes[:, 0] * (ix1 - ix0) + ix0
  yc = boxes[:, 1] * (iy1 - iy0) + iy0
  w = boxes[:, 2] * (ix1 - ix0)
  h = boxes[:, 3] * (iy1 - iy0)
  return torch.stack([xc, yc, w, h], dim=1)

def norms_to_indices(boxes,H,W=None):
    if W is None:
        W=H
    x0,x1 = boxes[:,0]*(W-1),boxes[:,2]*(W-1)+1
    y0,y1 = boxes[:,1]*(H-1),boxes[:,3]*(H-1)+1
    return torch.stack([x0, y0, x1, y1], dim=1).round().long()

def apply_box_transform(anchors, transforms):
  """
  Apply box transforms to a set of anchor boxes.

  Inputs:
  - anchors: Anchor boxes of shape (N, 4), where each anchor is specified
    in the form [xc, yc, w, h]
  - transforms: Box transforms of shape (N, 4) where each transform is
    specified as [tx, ty, tw, th]

  Returns:
  - boxes: Transformed boxes of shape (N, 4) where each box is in the
    format [xc, yc, w, h]
  """
  # Unpack anchors
  xa, ya = anchors[:, 0], anchors[:, 1]
  wa, ha = anchors[:, 2], anchors[:, 3]

  # Unpack transforms
  tx, ty = transforms[:, 0], transforms[:, 1]
  tw, th = transforms[:, 2], transforms[:, 3]

  x = xa + tx * wa
  y = ya + ty * ha
  w = wa * tw.exp()
  h = ha * th.exp()

  boxes = torch.stack([x, y, w, h], dim=1)
  return boxes


def invert_box_transform(anchors, boxes):
  """
  Compute the box transform that, when applied to anchors, would give boxes.

  Inputs:
  - anchors: Box anchors of shape (N, 4) in the format [xc, yc, w, h]
  - boxes: Target boxes of shape (N, 4) in the format [xc, yc, w, h]

  Returns:
  - transforms: Box transforms of shape (N, 4) in the format [tx, ty, tw, th]
  """
  # Unpack anchors
  xa, ya = anchors[:, 0], anchors[:, 1]
  wa, ha = anchors[:, 2], anchors[:, 3]
  
  # Unpack boxes
  x, y = boxes[:, 0], boxes[:, 1]
  w, h = boxes[:, 2], boxes[:, 3]

  tx = (x - xa) / wa
  ty = (y - ya) / ha
  tw = w.log() - wa.log()
  th = h.log() - ha.log()

  transforms = torch.stack([tx, ty, tw, th], dim=1)
  return transforms


def centers_to_extents(boxes):
  """
  Convert boxes from [xc, yc, w, h] format to [x0, y0, x1, y1] format

  Input:
  - boxes: Input boxes of shape (N, 4) in [xc, yc, w, h] format

  Returns:
  - boxes: Output boxes of shape (N, 4) in [x0, y0, x1, y1] format
  """
  xc, yc = boxes[:, 0], boxes[:, 1]
  w, h = boxes[:, 2], boxes[:, 3]

  x0 = xc - w / 2
  x1 = x0 + w
  y0 = yc - h / 2
  y1 = y0 + h

  boxes_out = torch.stack([x0, y0, x1, y1], dim=1)
  return boxes_out


def extents_to_centers(boxes):
  """
  Convert boxes from [x0, y0, x1, y1] format to [xc, yc, w, h] format

  Input:
  - boxes: Input boxes of shape (N, 4) in [x0, y0, x1, y1] format

  Returns:
  - boxes: Output boxes of shape (N, 4) in [xc, yc, w, h] format
  """
  x0, y0 = boxes[:, 0], boxes[:, 1]
  x1, y1 = boxes[:, 2], boxes[:, 3]

  xc = 0.5 * (x0 + x1)
  yc = 0.5 * (y0 + y1)
  w = x1 - x0
  h = y1 - y0

  boxes_out = torch.stack([xc, yc, w, h], dim=1)
  return boxes_out

