#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licens8.0es/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign

from . import box_utils
from .graph import GraphTripleConv, GraphTripleConvNet
from .layout import boxes_to_layout, masks_to_layout, boxes_to_seg, masks_to_seg
from .layers import build_mlp,build_cnn
from .utils import vocab

class Model(nn.Module):
  def __init__(self,
              embedding_dim=128,
              image_size=(128,128),
              input_dim=3,
              attribute_dim=35,
              # graph_net
              gconv_dim=128,
              gconv_hidden_dim=512,
              gconv_num_layers=5,
              # inside_cnn
              inside_cnn_arch="C3-32-2,C3-64-2,C3-128-2,C3-256-2",
              # refinement_net
              refinement_dims=(1024, 512, 256, 128, 64),
              # box_refine
              box_refine_arch = "I15,C3-64-2,C3-128-2,C3-256-2",
              roi_output_size = (8,8),
              roi_spatial_scale = 1.0/8.0,
              roi_cat_feature = True,
              # others
              mlp_activation='leakyrelu',
              mlp_normalization='none',
              cnn_activation='leakyrelu',
              cnn_normalization='batch'
              ):
    super(Model, self).__init__()
    ''' embedding '''
    self.vocab = vocab
    num_objs = len(vocab['object_idx_to_name'])
    num_preds = len(vocab['pred_idx_to_name'])
    num_doors = len(vocab['door_idx_to_name'])
    self.obj_embeddings = nn.Embedding(num_objs, embedding_dim)
    self.pred_embeddings = nn.Embedding(num_preds, embedding_dim)
    self.image_size = image_size
    self.feature_dim = embedding_dim+attribute_dim

    ''' graph_net '''
    self.gconv = GraphTripleConv(
      embedding_dim,
      attributes_dim=attribute_dim, 
      output_dim=gconv_dim,
      hidden_dim=gconv_hidden_dim,
      mlp_normalization=mlp_normalization
    )
    self.gconv_net = GraphTripleConvNet(
      gconv_dim,
      num_layers=gconv_num_layers-1,
      mlp_normalization=mlp_normalization
    )
  
    ''' inside_cnn '''
    inside_cnn,inside_feat_dim = build_cnn(
        f'I{input_dim},{inside_cnn_arch}',
        padding='valid'
    )
    self.inside_cnn = nn.Sequential(
      inside_cnn,
      nn.AdaptiveAvgPool2d(1)
    )
    inside_output_dim = inside_feat_dim
    obj_vecs_dim = gconv_dim+inside_output_dim

    ''' box_net '''
    box_net_dim = 4
    box_net_layers = [obj_vecs_dim, gconv_hidden_dim, box_net_dim]
    self.box_net = build_mlp(
      box_net_layers,
      activation=mlp_activation, 
      batch_norm=mlp_normalization
    )
    
    ''' relationship_net '''
    rel_aux_layers = [obj_vecs_dim, gconv_hidden_dim, num_doors]
    self.rel_aux_net = build_mlp(
      rel_aux_layers,
      activation=mlp_activation, 
      batch_norm=mlp_normalization
    )

    ''' refinement_net '''
    if refinement_dims!=None:
      self.refinement_net,_ = build_cnn(f"I{obj_vecs_dim},C3-128,C3-64,C3-{num_objs}")
    else:
      self.refinement_net = None

    ''' roi '''
    self.box_refine_backbone = None
    self.roi_cat_feature = roi_cat_feature
    if box_refine_arch!=None:
      box_refine_cnn,box_feat_dim = build_cnn(
        box_refine_arch,
        padding='valid'
      )
      self.box_refine_backbone = box_refine_cnn
      self.roi_align = RoIAlign(roi_output_size,roi_spatial_scale,-1) #(256,8,8)
      self.down_sample = nn.AdaptiveAvgPool2d(1)
      box_refine_layers = [obj_vecs_dim+256 if self.roi_cat_feature else 256, 512, 4]
      self.box_reg =build_mlp(
          box_refine_layers,
          activation=mlp_activation, 
          batch_norm=mlp_normalization
      )

  def forward(
    self, 
    objs, 
    triples, 
    boundary,
    obj_to_img=None,
    attributes=None,
    boxes_gt=None, 
    generate=False,
    refine=False,
    relative=False,
    inside_box=None
    ):
    """
    Required Inputs:
    - objs: LongTensor of shape (O,) giving categories for all objects
    - triples: LongTensor of shape (T, 3) where triples[t] = [s, p, o]
      means that there is a triple (objs[s], p, objs[o])

    Optional Inputs:
    - obj_to_img: LongTensor of shape (O,) where obj_to_img[o] = i
      means that objects[o] is an object in image i. If not given then
      all objects are assumed to belong to the same image.
    - boxes_gt: FloatTensor of shape (O, 4) giving boxes to use for computing
      the spatial layout; if not given then use predicted boxes.
    """
    # input size
    O, T = objs.size(0), triples.size(0)
    s, p, o = triples.chunk(3, dim=1)           # All have shape (T, 1)
    s, p, o = [x.squeeze(1) for x in [s, p, o]] # Now have shape (T,)
    edges = torch.stack([s, o], dim=1)          # Shape is (T, 2)
    B = boundary.size(0)
    H, W = self.image_size
  
    if obj_to_img is None:
      obj_to_img = torch.zeros(O, dtype=objs.dtype, device=objs.device)
    
    ''' embedding '''
    obj_vecs = self.obj_embeddings(objs)
    pred_vecs = self.pred_embeddings(p)

    ''' attribute '''
    if attributes is not None:
      obj_vecs = torch.cat([obj_vecs,attributes],1)
    obj_vecs_orig = obj_vecs
    
    ''' gconv '''
    obj_vecs, pred_vecs = self.gconv(obj_vecs, pred_vecs, edges)
    obj_vecs, pred_vecs = self.gconv_net(obj_vecs, pred_vecs, edges)

    ''' inside '''
    inside_vecs = self.inside_cnn(boundary).view(B,-1)
    obj_vecs = torch.cat([obj_vecs,inside_vecs[obj_to_img]],dim=1)

    ''' box '''
    boxes_pred = self.box_net(obj_vecs)
    if relative: boxes_pred = box_utils.box_rel2abs(boxes_pred,inside_box,obj_to_img)

    ''' relation '''
    # unused, for door position predition
    # rel_scores = self.rel_aux_net(obj_vecs)

    ''' generate '''
    gene_layout = None
    boxes_refine = None
    layout_boxes = boxes_pred if boxes_gt is None else boxes_gt
    if generate:
      layout_features = boxes_to_layout(obj_vecs,layout_boxes,obj_to_img,H,W)
      gene_layout = self.refinement_net(layout_features)
      
    ''' box refine '''
    if refine:
      gene_feat = self.box_refine_backbone(gene_layout)
      rois = torch.cat([
        obj_to_img.float().view(-1,1),
        box_utils.centers_to_extents(layout_boxes)*H
      ],-1)
      roi_feat = self.down_sample(self.roi_align(gene_feat,rois)).flatten(1)
      roi_feat = torch.cat([
        roi_feat,
        obj_vecs
      ],-1)
      boxes_refine = self.box_reg(roi_feat)
      if relative: boxes_refine = box_utils.box_rel2abs(boxes_refine,inside_box,obj_to_img)

    return boxes_pred, gene_layout, boxes_refine
