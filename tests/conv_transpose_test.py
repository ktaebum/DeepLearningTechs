import unittest

import torch
import torch.nn as nn

import numpy as np

from utils.deconv import ConvTranspose2D

TOL = 1e-5  # tolerance


class DeconvTest(unittest.TestCase):

    def test_conv_transpose2d_no_pad_no_stride(self):

        x = torch.randn(1, 10, 14, 14)

        correct = nn.ConvTranspose2d(10, 5, 3)
        my = ConvTranspose2D(10, 5, 3)

        my.weight = correct.weight.data.numpy()
        my.bias = correct.bias.data.numpy()

        correct = correct(x)
        my = my(x)

        self.assertLessEqual(
            np.linalg.norm(correct.detach().numpy() - my), TOL)

    def test_conv_transpose2d_no_pad_yes_stride(self):

        x = torch.randn(1, 10, 14, 14)

        correct = nn.ConvTranspose2d(10, 5, 3, stride=2)
        my = ConvTranspose2D(10, 5, 3, stride=2)

        my.weight = correct.weight.data.numpy()
        my.bias = correct.bias.data.numpy()

        correct = correct(x)
        my = my(x)

        self.assertLessEqual(
            np.linalg.norm(correct.detach().numpy() - my), TOL)

    def test_conv_transpose2d_yes_pad_no_stride(self):

        x = torch.randn(1, 10, 14, 14)

        correct = nn.ConvTranspose2d(10, 5, 3, padding=1)
        my = ConvTranspose2D(10, 5, 3, padding=1)

        my.weight = correct.weight.data.numpy()
        my.bias = correct.bias.data.numpy()

        correct = correct(x)
        my = my(x)

        self.assertLessEqual(
            np.linalg.norm(correct.detach().numpy() - my), TOL)

    def test_conv_transpose2d_yes_pad_yes_stride(self):

        x = torch.randn(1, 10, 14, 14)

        correct = nn.ConvTranspose2d(10, 5, 3, padding=1, stride=2)
        my = ConvTranspose2D(10, 5, 3, padding=1, stride=2)

        my.weight = correct.weight.data.numpy()
        my.bias = correct.bias.data.numpy()

        correct = correct(x)
        my = my(x)

        self.assertLessEqual(
            np.linalg.norm(correct.detach().numpy() - my), TOL)
