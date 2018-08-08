"""
File: deconv.py
Author: Taebum Kim
Email: phya.ktaebum@gmail.com
Github: https://github.com/ktaebum
Description:
    Numpy implementation for MaxUnpooling2D and ConvTranspose
"""
import numpy as np


class MaxUnpool2D:

    def __init__(self, kernel_size, stride=None, padding=0):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if stride is None:
            stride = kernel_size
        elif isinstance(stride, int):
            stride = (stride, stride)

        if isinstance(padding, int):
            padding = (padding, padding)

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def __call__(self, *args, **kwargs):
        return (self._forward(*args, **kwargs))

    def _forward(self, x, indices, output_size=None):
        B, C, HI, WI = x.shape

        PH, PW = self.padding
        SH, SW = self.stride
        HF, WF = self.kernel_size

        HO = SH * (HI - 1) - 2 * PH + HF  # cutoff padding later
        WO = SW * (WI - 1) - 2 * PW + WF

        out_shape = (B, C, HO, WO)

        out = np.zeros(out_shape, dtype=np.float32)

        for b in range(B):
            for c in range(C):
                idx_vector = indices[b, c].reshape(-1)
                x_vector = x[b, c].reshape(-1)

                out_vector = out[b, c].reshape(-1)
                out_vector[idx_vector] = x_vector

        return out


class ConvTranspose2D:

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=True):
        """
        ignored output_padding, groups, dilation
        implemented just forward pass
        """
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if isinstance(stride, int):
            stride = (stride, stride)

        if isinstance(padding, int):
            padding = (padding, padding)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = np.random.randn(
            in_channels,
            out_channels,
            *kernel_size,
        )
        self.bias = None
        if bias:
            self.bias = np.zeros(out_channels, dtype=np.float32)

    def __str__(self):
        result = 'ConvTranspose2D(%d, %d, kernel_size=%s, stride=%s)' % (
            self.in_channels, self.out_channels, str(self.kernel_size),
            str(self.stride))

        return result

    def __repr__(self):
        return str(self)

    def __call__(self, *args, **kwargs):
        return (self._forward(*args, **kwargs))

    def _forward(self, x):
        """
        x has shape (batch_size, in_channels, height, width)
        """
        B, C, HI, WI = x.shape  # input shape
        if C != self.in_channels:
            raise ValueError('Input must has %d channels' % self.in_channels)

        weight = self.weight
        _, O, HF, WF = weight.shape  # filter shape
        PH, PW = self.padding
        SH, SW = self.stride
        """
        Assume Output shape is B, O, HO, WO
        Then, since
        (HO + 2 * Padding - HF) // Stride + 1 == HI
        We can calculate output shape inversely
        """
        HO = SH * (HI - 1) + HF  # cutoff padding later
        WO = SW * (WI - 1) + WF

        output_shape = (B, O, HO, WO)

        out = np.zeros(output_shape, dtype=np.float32)

        for b in range(B):
            # for each minibatch
            for h in range(HI):
                for w in range(WI):
                    for o in range(O):
                        for c in range(C):
                            out[b, o, h * SH:h * SH + HF, w * SW:w * SW +
                                WF] += weight[c, o, :, :] * x[b, c, h, w]

        if self.bias is not None:
            out += self.bias.reshape(1, O, 1, 1)

        if self.padding[0] != 0:
            real_out = np.zeros(
                (B, O, HO - 2 * PH, WO - 2 * PW), dtype=np.float32)
            # cutoff padding
            for b in range(B):
                for o in range(O):
                    real_out[b, o] = out[b, o, PH:-PH, PW:-PW]
            return real_out
        else:
            return out
