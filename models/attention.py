import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from torchvision.models import vgg19


class AttentionEncoder(nn.Module):

    def __init__(self):
        super(AttentionEncoder, self).__init__()

        vggnet = vgg19(True)

        # get conv part except last pooling layer
        modules = list(vggnet.children())[0][:-1]

        for m in modules:
            if isinstance(m, nn.MaxPool2d):
                # change maxpool to return indices
                m.return_indices = True

        self.vggnet = nn.Sequential(*modules)
        self.switches = []

    def forward(self, x, store_switch=False):
        with torch.no_grad():
            out = x
            self.switches = []
            for layer in self.vggnet:
                if isinstance(layer, nn.MaxPool2d):
                    if store_switch:
                        out, idx = layer(out)
                        self.switches.append(idx)
                    else:
                        out, _ = layer(out)
                else:
                    out = layer(out)

            out = out.reshape(out.shape[0], 196, 512)
            return out

    def deconv(self, attention):
        switches = iter(reversed(self.switches))

        for layer in reversed(self.vggnet):
            if isinstance(layer, nn.ReLU):
                continue
            kernel_size = layer.kernel_size
            stride = layer.stride
            padding = layer.padding

            if isinstance(layer, nn.Conv2d):
                in_channels = layer.in_channels
                out_channels = layer.out_channels
                dilation = layer.dilation
                groups = layer.groups

                transpose = nn.ConvTranspose2d(
                    in_channels=out_channels,
                    out_channels=in_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    bias=False)

                transpose.weight.data = layer.weight.data

                attention = transpose(attention)
            else:
                unpool = nn.MaxUnpool2d(
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )

                switch = switches.__next__()
                attention = F.relu(unpool(attention, switch))

        return attention


class AttentionDecoder(nn.Module):

    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(AttentionDecoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTMCell(512 + embed_size, hidden_size, num_layers)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.dropout = nn.Dropout(0.5)

        # self.inith = nn.Linear(512, self.hidden_size)
        # self.initc = nn.Linear(512, self.hidden_size)

        self.fc_out = nn.Linear(hidden_size, vocab_size)
        self.fc_dropout = nn.Dropout(0.5)

        # these are for attention
        self.h2a = nn.Linear(hidden_size, 512)
        self.f2a = nn.Linear(512, 512)
        self.a2a = nn.Linear(512, 1)

    def _get_attention(self, feature, h):
        """
        B-L-D
        feature: batch * 196 * 512
        h: batch * hidden_size
        """
        a_feature = self.f2a(feature)  # B * 196 * 512
        a_h = self.h2a(h).unsqueeze(1)  # B * 1 * 512

        a = F.relu(a_feature + a_h)  # B * 196 * 512
        a = self.a2a(a).squeeze(2)  # B * 196 * 1 => B * 196
        alpha = F.softmax(a, 1)  # B * 196
        context = torch.sum(feature * alpha.unsqueeze(2), 1)  # B * 512

        return context

    def _init_hidden(self, batch_size, device):
        h = Variable(torch.zeros(batch_size, self.hidden_size)).to(device)
        c = Variable(torch.zeros(batch_size, self.hidden_size)).to(device)
        return (h, c)

    def forward(self, feature, captions, lengths):
        """
        feature: batch_size * 196 * 512 attention feature
        """
        batch_size, timestep = captions.shape

        embedding = self.embedding(captions)
        embedding = self.dropout(embedding)
        # batch * seq_len * embed_size

        z = torch.mean(feature, 1)  # batch * 512

        # h_0 = self.inith(z)
        # c_0 = self.initc(z)
        h_0, c_0 = self._init_hidden(batch_size, feature.device)

        predicts = torch.zeros(batch_size, timestep,
                               self.vocab_size).to(feature.device)

        for t in range(timestep):
            batch_size = sum(i >= t for i in lengths)  # effective batch
            if t != 0:
                z = self._get_attention(feature[:batch_size],
                                        h_0[:batch_size])

            embedding_batch = embedding[:batch_size, t].squeeze(
                1)  # batch_size * embed_size
            batch_cat = torch.cat((z, embedding_batch),
                                  1)  # batch_size * (embed_size + 512)
            h_0, c_0 = self.lstm(batch_cat,
                                 (h_0[:batch_size], c_0[:batch_size]))

            output = self.fc_out(self.fc_dropout(h_0))
            predicts[:batch_size, t] = output

        return predicts

    def sample(self, feature, sentence_len=20):
        sampled_id = []
        batch_size = feature.shape[0]

        z = torch.mean(feature, 1)

        h_0, c_0 = self._init_hidden(batch_size, feature.device)
        words = self.embedding(torch.ones(batch_size,
                                          1).long().cuda()).squeeze(1)

        attentions = [z]
        for s in range(sentence_len):
            if s != 0:
                z = self._get_attention(feature, h_0)
                attentions.append(z)

            cat = torch.cat([z, words], 1)
            h_0, c_0 = self.lstm(cat, (h_0, c_0))

            outputs = self.fc_out(h_0)

            _, predicted = torch.max(outputs, 1)

            sampled_id.append(predicted.unsqueeze(1))
            words = self.embedding(predicted)

        sampled_id = torch.cat(sampled_id, 1)
        return sampled_id.squeeze(), attentions
