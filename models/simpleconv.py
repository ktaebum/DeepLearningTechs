import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleConv(nn.Module):
    """
    Simple convolutional layer for CIFAR-10
    8-layer
    Use for deconvolution
    """

    def __init__(self):
        super(SimpleConv, self).__init__()

        self.layers = nn.ModuleList()

        # layer 1
        self.layers.append(
            nn.Sequential(
                nn.Conv2d(3, 96, 5),  # 96 x 28 x 28
                nn.ReLU(),
                nn.MaxPool2d(2, 2, return_indices=True),  # 96 x 14 x 14
            ))

        # layer 2
        self.layers.append(
            nn.Sequential(
                nn.Conv2d(96, 256, 3),  # 256 x 12 x 12
                nn.ReLU(),
                nn.MaxPool2d(2, 2, return_indices=True),  # 256 x 6 x 6
            ))

        # layer 3
        self.layers.append(
            nn.Sequential(
                nn.Conv2d(256, 384, 3, padding=1),  # 384 x 6 x 6
                nn.ReLU(),
            ))

        # layer 4
        self.layers.append(
            nn.Sequential(
                nn.Conv2d(384, 256, 3, padding=1),
                nn.ReLU(),
            ))

        # layer 5
        self.layers.append(
            nn.Sequential(
                nn.MaxPool2d(2, 2, return_indices=True),  # 256 x 3 x 3
            ))

        # layer 6-8
        self.fcs = nn.Sequential(
            nn.Linear(2304, 1000),
            nn.ReLU(),
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        out = x
        for layer in self.layers:
            for seq in layer:
                if not isinstance(seq, nn.MaxPool2d):
                    # just forward
                    out = seq(out)
                else:
                    # maxpool, discard index
                    out, _ = seq(out)

        out = out.view(out.shape[0], -1)
        out = self.fcs(out)
        return out

    def deconvolution(self, x, layer_num):
        """
        get deconvolution result from target layer_num
        layer_num is not zero-base indexing
        """

        if layer_num < 1 or layer_num > 5:
            raise ValueError('Please set layer_num between 1 to 5')

        switches = []
        origin_layers = []
        with torch.no_grad():
            out = x
            for i, layer in enumerate(self.layers, 1):
                for seq in layer:
                    if isinstance(seq, nn.ReLU):
                        out = seq(out)
                    elif isinstance(seq, nn.Conv2d):
                        out = seq(out)
                        origin_layers.append(seq)
                    else:
                        # max pool2d
                        out, idxs = seq(out)
                        switches.append(idxs)
                        origin_layers.append(seq)
                if i == layer_num:
                    break

            feature = out

            switch_iter = iter(reversed(switches))
            for layer in reversed(origin_layers):
                kernel_size = layer.kernel_size
                stride = layer.stride
                padding = layer.padding
                if isinstance(layer, nn.Conv2d):
                    in_channels = layer.in_channels
                    out_channels = layer.out_channels
                    dilation = layer.dilation
                    groups = layer.groups
                    # bias = layer.bias is not None

                    transpose = nn.ConvTranspose2d(
                        in_channels=out_channels,
                        out_channels=in_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        groups=groups,
                        bias=False,
                    )

                    transpose.weight.data = layer.weight.data

                    feature = transpose(feature)
                else:
                    # maxpool
                    unpool = nn.MaxUnpool2d(
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    )

                    switch = switch_iter.__next__()
                    feature = F.relu(unpool(feature, switch))
        return feature
