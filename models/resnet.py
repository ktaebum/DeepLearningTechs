"""
File: resnet.py
Author: Taebum Kim 
Email: phya.ktaebum@gmail.com
Github: https://github.com/ktaebum 
Description: 
    Implementation of Residual Network
    Since modified to fit cifar 10,
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from utils.train import ModelTrainer

# original version is (256, 512, 1024, 2048)
bottle_neck_channels = (64, 128, 256, 512, 1024)
basic_channels = (64, 64, 128, 256, 512)

cfgs = {
    '18': (2, 2, 2, 2),
    '34': (3, 4, 6, 3),
    '50': (3, 4, 6, 3),
    '101': (3, 4, 23, 3),
    '152': (3, 8, 36, 3)
}

strides = (1, 1, 2, 2)


class ResNet(nn.Module):
    """
    For cifar10
    to fit 28 x 28 after conv2_x
    I modified structure as follows
    conv1: 64 3x3 conv, stride=1, padding = 1 (after this, images are 32x32)
    conv2: delete pooling layer, down sample at conv2_1
    conv3: down sample 30 -> 28
    rests are same
    """

    def __init__(self, layer_num='18'):
        super(ResNet, self).__init__()

        if layer_num not in cfgs.keys():
            raise KeyError('Invalid Mode: %s' % layer_num)

        # set basic building block
        use_basic = layer_num in ('18', '34')
        block = BasicBlock if use_basic else BottleNeckBlock
        channels = basic_channels if use_basic else bottle_neck_channels
        cfg = cfgs[layer_num]

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )

        self._mode = layer_num

        self.residuals = []
        for i in range(4):
            self.residuals.append(
                ResidualLayer(
                    block,
                    cfg[i],
                    channels[i],
                    channels[i + 1],
                    strides[i],
                ))

        self.residuals = nn.Sequential(*self.residuals)

        self.fc = nn.Linear(7 * 7 * channels[-1], 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.residuals(out)

        out = out.reshape(-1, np.prod(out.shape[1:]))
        out = self.fc(out)

        return out

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        raise RuntimeError('Cannot modify mode after construction')


class ResNetTrainer(ModelTrainer):

    def __init__(self, *args, **kwargs):
        super(ResNetTrainer, self).__init__(*args, **kwargs)

    def update_optimizer(self, outputs, labels):
        self.params['optimizer'].zero_grad()
        loss = self.calculate_loss(outputs, labels)
        loss.backward()
        self.params['optimizer'].step()
        return loss

    def calculate_loss(self, outputs, labels):
        loss = self.params['loss_function'](outputs, labels)
        return loss


class ResidualLayer(nn.Module):

    def __init__(self, block, stack, in_channel, out_channel, stride=2):
        super(ResidualLayer, self).__init__()
        self.block = block

        self.blocks = nn.ModuleList([])
        for s in range(stack):
            if s == 0:
                # down sample!
                self.blocks.append(
                    self.block(in_channel, out_channel, True, stride))
            else:
                # not down sample
                self.blocks.append(
                    self.block(out_channel, out_channel, False, 1))

        padding = 0 if stride == 1 else 1

        self.resize_input = nn.Sequential(
            nn.Conv2d(
                in_channel,
                out_channel,
                3,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channel),
        )

    def forward(self, x):
        out = x
        block_len = len(self.blocks)
        for s in range(block_len):
            resize = self.resize_input if s == 0 else nn.Sequential()
            shortcut = resize(out)
            conv_result = self.blocks[s](out)
            out = shortcut + conv_result
            out = F.relu(out)

        return out


class BottleNeckBlock(nn.Module):

    def __init__(self, in_channels, out_channels, downsample=False,
                 stride=1):
        super(BottleNeckBlock, self).__init__()

        if downsample:
            if stride == 1:
                padding = 0
            else:
                # stride == 2
                padding = 1
        else:
            padding = 1

        if in_channels == out_channels:
            # not the first block
            conv1_projection = in_channels // 4
        else:
            conv1_projection = in_channels // 2

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, conv1_projection, 1, 1, 0),
            nn.BatchNorm2d(conv1_projection),
            nn.ReLU(True),
            nn.Conv2d(
                conv1_projection,
                conv1_projection,
                3,
                stride,
                padding=padding),
            nn.BatchNorm2d(conv1_projection),
            nn.ReLU(True),
            nn.Conv2d(conv1_projection, out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.conv(x)


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, downsample=False,
                 stride=1):
        super(BasicBlock, self).__init__()

        if downsample:
            if stride == 1:
                padding = 0
            else:
                # stride == 2
                padding = 1
        else:
            padding = 1

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(
                out_channels,
                out_channels,
                3,
                stride=stride,
                padding=padding),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.conv(x)
