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

import functools

import torch
import torch.nn as nn
from torch.nn.functional import interpolate

class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins, BatchNorm):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                BatchNorm(reduction_dim),
                #nn.ReLU(inplace=True)
                nn.LeakyReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)

def get_normalization_2d(channels, normalization):
    if normalization == 'instance':
        return nn.InstanceNorm2d(channels)
    elif normalization == 'batch':
        return nn.BatchNorm2d(channels)
    elif normalization == 'none':
        return None
    else:
        raise ValueError('Unrecognized normalization type "%s"' % normalization)


def get_activation(name):
    kwargs = {}
    if name.lower().startswith('leakyrelu'):
        if '-' in name:
            slope = float(name.split('-')[1])
            kwargs = {'negative_slope': slope}
    name = 'leakyrelu'
    activations = {
        'relu': nn.ReLU,
        'leakyrelu': nn.LeakyReLU,
    }
    if name.lower() not in activations:
        raise ValueError('Invalid activation "%s"' % name)
    return activations[name.lower()](**kwargs)


def _init_conv(layer, method):
    if not isinstance(layer, nn.Conv2d):
        return
    if method == 'default':
        return
    elif method == 'kaiming-normal':
        nn.init.kaiming_normal(layer.weight)
    elif method == 'kaiming-uniform':
        nn.init.kaiming_uniform(layer.weight)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

    def __repr__(self):
        return 'Flatten()'


class Unflatten(nn.Module):
    def __init__(self, size):
        super(Unflatten, self).__init__()
        self.size = size

    def forward(self, x):
        return x.view(*self.size)

    def __repr__(self):
        size_str = ', '.join('%d' % d for d in self.size)
        return 'Unflatten(%s)' % size_str


class GlobalAvgPool(nn.Module):
    def forward(self, x):
        N, C = x.size(0), x.size(1)
        return x.view(N, C, -1).mean(dim=2)


class ResidualBlock(nn.Module):
    def __init__(self, channels, normalization='batch', activation='relu',
                 padding='same', kernel_size=3, init='default'):
        super(ResidualBlock, self).__init__()

        K = kernel_size
        P = _get_padding(K, padding)
        C = channels
        self.padding = P
        layers = [
            get_normalization_2d(C, normalization),
            get_activation(activation),
            nn.Conv2d(C, C, kernel_size=K, padding=P),
            get_normalization_2d(C, normalization),
            get_activation(activation),
            nn.Conv2d(C, C, kernel_size=K, padding=P),
        ]
        layers = [layer for layer in layers if layer is not None]
        for layer in layers:
            _init_conv(layer, method=init)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        P = self.padding
        shortcut = x
        if P == 0:
            shortcut = x[:, :, P:-P, P:-P]
        y = self.net(x)
        return shortcut + self.net(x)


def _get_padding(K, mode):
    """ Helper method to compute padding size """
    if mode == 'valid':
        return 0
    elif mode == 'same':
        assert K % 2 == 1, 'Invalid kernel size %d for "same" padding' % K
        return (K - 1) // 2


def build_cnn(arch, normalization='batch', activation='leakyrelu', padding='same',
              pooling='max', init='default'):
    """
    Build a CNN from an architecture string, which is a list of layer
    specification strings. The overall architecture can be given as a list or as
    a comma-separated string.

    All convolutions *except for the first* are preceeded by normalization and
    nonlinearity.

    All other layers support the following:
    - IX: Indicates that the number of input channels to the network is X.
          Can only be used at the first layer; if not present then we assume
          3 input channels.
    - CK-X: KxK convolution with X output channels
    - CK-X-S: KxK convolution with X output channels and stride S
    - R: Residual block keeping the same number of channels
    - UX: Nearest-neighbor upsampling with factor X
    - PX: Spatial pooling with factor X
    - FC-X-Y: Flatten followed by fully-connected layer

    Returns a tuple of:
    - cnn: An nn.Sequential
    - channels: Number of output channels
    """
    if isinstance(arch, str):
        arch = arch.split(',')
    cur_C = 3
    if len(arch) > 0 and arch[0][0] == 'I':
        cur_C = int(arch[0][1:])
        arch = arch[1:]

    first_conv = True
    flat = False
    layers = []
    for i, s in enumerate(arch):
        if s[0] == 'C':
            if not first_conv:
                layers.append(get_normalization_2d(cur_C, normalization))
                layers.append(get_activation(activation))
            first_conv = False
            vals = [int(i) for i in s[1:].split('-')]
            if len(vals) == 2:
                K, next_C = vals
                stride = 1
            elif len(vals) == 3:
                K, next_C, stride = vals
            # K, next_C = (int(i) for i in s[1:].split('-'))
            P = _get_padding(K, padding)
            conv = nn.Conv2d(cur_C, next_C, kernel_size=K, padding=P, stride=stride)
            layers.append(conv)
            _init_conv(layers[-1], init)
            cur_C = next_C
        elif s[0] == 'R':
            norm = 'none' if first_conv else normalization
            res = ResidualBlock(cur_C, normalization=norm, activation=activation,
                                padding=padding, init=init)
            layers.append(res)
            first_conv = False
        elif s[0] == 'U':
            factor = int(s[1:])
            layers.append(Interpolate(scale_factor=factor, mode='nearest'))
        elif s[0] == 'P':
            factor = int(s[1:])
            if pooling == 'max':
                pool = nn.MaxPool2d(kernel_size=factor, stride=factor)
            elif pooling == 'avg':
                pool = nn.AvgPool2d(kernel_size=factor, stride=factor)
            layers.append(pool)
        elif s[:2] == 'FC':
            _, Din, Dout = s.split('-')
            Din, Dout = int(Din), int(Dout)
            if not flat:
                layers.append(Flatten())
            flat = True
            layers.append(nn.Linear(Din, Dout))
            if i + 1 < len(arch):
                layers.append(get_activation(activation))
            cur_C = Dout
        else:
            raise ValueError('Invalid layer "%s"' % s)
    layers = [layer for layer in layers if layer is not None]
    # for layer in layers:
    #   print(layer)
    return nn.Sequential(*layers), cur_C


def build_mlp(dim_list, activation='leakyrelu', batch_norm='none',
              dropout=0, final_nonlinearity=True):
    layers = []
    for i in range(len(dim_list) - 1):
        dim_in, dim_out = dim_list[i], dim_list[i + 1]
        layers.append(nn.Linear(dim_in, dim_out))
        final_layer = (i == len(dim_list) - 2)
        if not final_layer or final_nonlinearity:
            if batch_norm == 'batch':
                layers.append(nn.BatchNorm1d(dim_out))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leakyrelu':
                layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super(ConditionalBatchNorm2d).__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'conditional':
        norm_layer = functools.partial(ConditionalBatchNorm2d)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super(Interpolate, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                           align_corners=self.align_corners)
