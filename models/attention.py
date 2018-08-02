import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import vgg19_bn
from preprocess.nlp import Vocabulary


class AttentionEncoder(nn.Module):

    def __init__(self):
        super(AttentionEncoder, self).__init__()

        vggnet = vgg19_bn(True)

        # get conv part except last pooling layer
        modules = list(vggnet.children())[0][:-1]

        self.vggnet = nn.Sequential(*modules)

    def forward(self, x):
        with torch.no_grad():
            output = self.vggnet(x)

        output = output.reshape(output.shape[0], 196, 512)
        return output


class AttentionDecoder(nn.Module):

    def __init__(self, embed_size, hidden_size, vocab_size, num_layer=1):
        super(AttentionDecoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        # self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.lstm = nn.LSTMCell(512 + embed_size, hidden_size)

        self.vocab_size = vocab_size

        self.initc = nn.Linear(512, hidden_size)
        self.inith = nn.Linear(512, hidden_size)

        self.Lo = nn.Linear(embed_size, vocab_size)
        self.Lh = nn.Linear(hidden_size, embed_size)
        self.Lz = nn.Linear(512, embed_size)

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
        feature = torch.sum(feature * alpha.unsqueeze(2), 1)  # B * 512

        return feature

    def forward(self, feature, captions, lengths):
        """
        feature: batch_size * 196 * 512 attention feature
        """
        batch_size, timestep = captions.shape

        embedding = self.embedding(captions)  # batch * seq_len * embed_size

        feature_mean = torch.mean(feature, 1)  # batch * 512
        h_0 = self.inith(feature_mean)
        c_0 = self.initc(feature_mean)

        predicts = torch.zeros(batch_size, timestep, self.vocab_size)

        for t in range(timestep):
            batch_size = sum(i >= t for i in lengths)  # effective batch
            z_batch = self._get_attention(feature[:batch_size],
                                          h_0[:batch_size])

            embedding_batch = embedding[:batch_size, t].squeeze(
                1)  # batch_size * embed_size
            batch_cat = torch.cat((z_batch, embedding_batch),
                                  1)  # batch_size * (embed_size + 512)
            h_0, c_0 = self.lstm(batch_cat,
                                 (h_0[:batch_size], c_0[:batch_size]))

            predicts[:batch_size, t] = self.Lo(
                embedding_batch + self.Lh(h_0) + self.Lz(z_batch))

        return predicts

    def sample(self, start_input, feature, sentence_len=20):
        sampled_id = []

        feature_mean = torch.mean(feature, 1)
        embedding = self.embedding(start_input)

        h_0 = self.inith(feature_mean)
        c_0 = self.initc(feature_mean)

        for s in range(sentence_len):
            z = self._get_attention(feature, h_0)

            cat = torch.cat([z, embedding], 1)
            h_0, c_0 = self.lstm(cat, (h_0, c_0))

            outputs = self.Lo(embedding + self.Lh(h_0) + self.Lz(z))

            _, predicted = torch.max(outputs, 1)

            sampled_id.append(predicted.unsqueeze(1))
            embedding = self.embedding(predicted)

        sampled_id = torch.cat(sampled_id, 1)
        return sampled_id.squeeze()

        return
