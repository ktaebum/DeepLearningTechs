import unittest

import torch
import torch.nn as nn

import numpy as np

from utils.deconv import MaxUnpool2D

TOL = 1e-5  # tolerance


class UnpoolTest(unittest.TestCase):

    def test_unpool2d_no_pad_no_stride(self):

        x = torch.randn(1, 3, 14, 14)

        forward = nn.MaxPool2d(2, 1, padding=0, return_indices=True)

        out, idx = forward(x)

        correct = nn.MaxUnpool2d(2, 1, padding=0)
        my = MaxUnpool2D(2, 1, padding=0)

        correct = correct(out, idx)
        my = my(out, idx)

        self.assertLessEqual(
            np.linalg.norm(correct.detach().numpy() - my), TOL)

    def test_unpool2d_no_pad_yes_stride(self):

        x = torch.randn(1, 3, 14, 14)

        forward = nn.MaxPool2d(2, 2, padding=0, return_indices=True)

        out, idx = forward(x)

        correct = nn.MaxUnpool2d(2, 2, padding=0)
        my = MaxUnpool2D(2, 2, padding=0)

        correct = correct(out, idx)
        my = my(out, idx)

        self.assertLessEqual(
            np.linalg.norm(correct.detach().numpy() - my), TOL)

    def test_unpool2d_yes_pad_no_stride(self):

        x = torch.randn(1, 3, 14, 14)

        forward = nn.MaxPool2d(2, 1, padding=1, return_indices=True)

        out, idx = forward(x)

        correct = nn.MaxUnpool2d(2, 1, padding=1)
        my = MaxUnpool2D(2, 1, padding=1)

        correct = correct(out, idx)
        my = my(out, idx)

        self.assertLessEqual(
            np.linalg.norm(correct.detach().numpy() - my), TOL)
