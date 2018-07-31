import torch
import torch.nn as nn
import torch.nn.functional as F

_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SingleLSTM(nn.Module):
    """
    Single Layer LSTM
    """

    def __init__(self, input_size, hidden_size, bias=True):
        """
        input_size: dim of input_size
        hidden_size: dim of hidden_size
        bias: whether activates bias
        """
        self.input_size = input_size
        self.hidden_size = hidden_size

        gate_size = 4 * hidden_size

        self.i2h = nn.Linear(input_size, gate_size, bias=False)
        self.h2h = nn.Linear(gate_size, gate_size, bias=False)

        self.bias = None
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(gate_size), requires_grad=True)

    def forward(self, x, h, c):
        """
        Single Timestep forward
        x is shape of (batch_size, input_dim)
        """
        stacked = self.i2h(x) + self.h2h(h)
        if self.bias is not None:
            stacked += self.bias

        ii, ff, oo, cc = torch.split(stacked, self.hidden_size, 1)

        i = F.relu(ii)
        f = F.relu(ff)
        o = F.relu(oo)
        c = f * c + i * F.tanh(cc)

        h = o * c

        return h, c

    def init_h_c(self, batch_size):
        return (
            torch.zeros(batch_size, self.hidden_size).to(_device),
            torch.zeros(batch_size, self.hidden_size).to(_device),
        )
