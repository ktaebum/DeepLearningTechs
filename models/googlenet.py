# -*- coding: utf-8 -*-
"""
File: googlenet.py
Author: Taebum Kim
Email: phya.ktaebum@gmail.com
Github: https://github.com/ktaebum
Description:
    Implementation of googlenet
    Origin paper: https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf
    Modified version for cifar10
"""

import torch
import torch.nn as nn

from utils.train import ModelTrainer

VALID_MODES = ('naive', 'improved')

# naive inception configuration
cfg_naive = {
    '3': ((192, 64, 128, 32, 32), (256, 128, 192, 96, 64)),
    '4': ((480, 192, 208, 48, 64), (512, 160, 224, 64, 64),
          (512, 128, 256, 64, 64), (512, 112, 288, 64, 64), (528, 256, 320,
                                                             128, 128)),
    '5': ((832, 256, 320, 128, 128), (832, 384, 384, 128, 128))
}

cfg_improved = {
    '3': ((192, 64, 96, 128, 16, 32, 32), (256, 128, 128, 192, 32, 96, 64)),
    '4': ((480, 192, 96, 208, 16, 48, 64), (512, 160, 112, 224, 24, 64, 64),
          (512, 128, 128, 256, 24, 64, 64), (512, 112, 144, 288, 32, 64, 64),
          (528, 256, 160, 320, 32, 128, 128)),
    '5': ((832, 256, 160, 320, 32, 128, 128), (832, 384, 192, 384, 48, 128,
                                               128))
}


class InceptionModuleNaive(nn.Module):
    """
    Naive Inception Module
    """

    def __init__(self, in_channel, _1x1, _3x3, _5x5, pool_proj):
        """
        in_channel: input's channel
        _1x1: output channel of 1x1 convolutions (leftmost block)
        _3x3: output channel of 3x3 conv (second block)
        _5x5: output channel of 5x5 conv (third block)
        pool_proj: output channel of 1x1 after 3x3 max pool (rightmost block)
        """
        super(InceptionModuleNaive, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channel, _1x1, 1), nn.BatchNorm2d(_1x1),
            nn.ReLU(True))

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channel, _3x3, 3, padding=1), nn.BatchNorm2d(_3x3),
            nn.ReLU(True))

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channel, _5x5, 5, padding=2), nn.BatchNorm2d(_5x5),
            nn.ReLU(True))

        self.block4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channel, pool_proj, 1), nn.BatchNorm2d(pool_proj),
            nn.ReLU(True))

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(x)
        out3 = self.block3(x)
        out4 = self.block4(x)
        return torch.cat((out1, out2, out3, out4), 1)


class InceptionModuleImproved(nn.Module):
    """
    Improved (dimensionality reduction) Inception Module
    """

    def __init__(self, in_channel, _1x1, _3x3reduce, _3x3, _5x5reduce, _5x5,
                 pool_proj):
        """
        in_channel: input's channel
        _1x1: output channel of 1x1 convolutions (leftmost block)
        _3x3reduce: output channel of 1x1 conv before 3x3 conv
        _3x3: output channel of 3x3 conv (second block)
        _5x5reduce: output channel of 1x1 conv before 5x5 conv
        _5x5: output channel of 5x5 conv (third block)
        pool_proj: output channel of 1x1 after 3x3 max pool (rightmost block)
        """
        super(InceptionModuleImproved, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channel, _1x1, 1), nn.BatchNorm2d(_1x1),
            nn.ReLU(True))

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channel, _3x3reduce, 1), nn.BatchNorm2d(_3x3reduce),
            nn.ReLU(True), nn.Conv2d(_3x3reduce, _3x3, 3, padding=1),
            nn.BatchNorm2d(_3x3), nn.ReLU(True))

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channel, _5x5reduce, 1), nn.BatchNorm2d(_5x5reduce),
            nn.ReLU(True), nn.Conv2d(_5x5reduce, _5x5, 5, padding=2),
            nn.BatchNorm2d(_5x5), nn.ReLU(True))

        self.block4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channel, pool_proj, 1), nn.BatchNorm2d(pool_proj),
            nn.ReLU(True))

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(x)
        out3 = self.block3(x)
        out4 = self.block4(x)
        return torch.cat((out1, out2, out3, out4), 1)


class GoogleNet(nn.Module):
    """
    GoogleNet
    """

    def __init__(self, mode='naive', aux=True):

        super(GoogleNet, self).__init__()
        self.stem = nn.Sequential(
            # first stem
            # They do not use inception in very first layers
            # Modified to fit (28 * 28 * 192) after stem
            # (in our case, it is cifar-10, not imagenet)
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 192, 5, 1),
            nn.ReLU(True))

        # mode setting
        self._mode = mode
        if mode == 'naive':
            self.Inception = InceptionModuleNaive
            cfg = cfg_naive
        elif mode == 'improved':
            self.Inception = InceptionModuleImproved
            cfg = cfg_improved
        else:
            raise ValueError(
                'Invalid mode %s, pick among {}'.format(VALID_MODES) % mode)

        self.aux = aux
        if aux:
            self.aux1_conv = nn.Sequential(
                nn.AvgPool2d(5, 3), nn.Conv2d(512, 64, 1))
            self.aux1_fc = nn.Sequential(
                nn.Linear(1024, 512), nn.Dropout(0.7), nn.Linear(512, 10))
            self.aux2_conv = nn.Sequential(
                nn.AvgPool2d(5, 3), nn.Conv2d(528, 64, 1))
            self.aux2_fc = nn.Sequential(
                nn.Linear(1024, 512), nn.Dropout(0.7), nn.Linear(512, 10))

        self.inception3 = self._build_inception_layer(cfg['3'])
        self.pool3 = nn.MaxPool2d(3, 2, 1)
        self.inception4 = self._build_inception_layer(cfg['4'])
        self.pool4 = nn.MaxPool2d(3, 2, 1)
        self.inception5 = self._build_inception_layer(cfg['5'])
        self.pool5 = nn.AvgPool2d(7, 7, 1)

        self.fcs = nn.Sequential(
            # fully connected layer
            nn.Dropout(0.4, inplace=True),
            nn.Linear(1024, 512),
            nn.Linear(512, 10))

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        raise ValueError('You cannot modify mode after construction')

    def forward(self, x):
        out = self.stem(x)
        out = self.inception3(out)
        out = self.pool3(out)
        aux1 = None
        aux2 = None
        if self.aux and self.training:
            # in training mode
            # add aux
            for i in range(len(self.inception4)):
                out = self.inception4[i](out)
                if i == 0:
                    # aux1 after 4a
                    aux1 = self.aux1_conv(out)
                    aux1 = aux1.reshape(aux1.shape[0], -1)
                    aux1 = self.aux1_fc(aux1)
                if i == 3:
                    # aux2 after 4d
                    aux2 = self.aux2_conv(out)
                    aux2 = aux2.reshape(aux2.shape[0], -1)
                    aux2 = self.aux2_fc(aux2)
        else:
            # in eval mode or not aux unit
            # just forward
            out = self.inception4(out)

        out = self.pool4(out)
        out = self.inception5(out)
        out = self.pool5(out)

        # flatten
        out = out.view(out.shape[0], -1)

        out = self.fcs(out)
        return out, aux1, aux2

    def _build_inception_layer(self, cfgs):
        """
        Build inception layer from configuration
        """
        layers = []
        for cfg in cfgs:
            layers.append(self.Inception(*cfg))
        return nn.Sequential(*layers)


class GoogleNetTrainer(ModelTrainer):

    def __init__(self, *args, **kwargs):
        super(GoogleNetTrainer, self).__init__(*args, **kwargs)

    def update_optimizer(self, outputs, labels):
        optimizer = self.params['optimizer']
        optimizer.zero_grad()
        loss = self.calculate_loss(outputs, labels)
        loss.backward()
        optimizer.step()
        return loss

    def calculate_loss(self, outputs, labels):
        out, aux1, aux2 = outputs
        loss_function = self.params['loss_function']

        loss = loss_function(out, labels)

        if aux1 is not None:
            loss += (0.3 * loss_function(aux1, labels))

        if aux2 is not None:
            loss += (0.3 * loss_function(aux2, labels))

        return loss

    def calculate_predicted_labels(self, outputs):
        out, _, _ = outputs

        _, predicts = torch.max(out, 1)
        return predicts

    pass
