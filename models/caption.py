import torch
import torch.nn as nn

from torch.backends import cudnn

from torchvision.models import resnet152
from utils.train import ModelTrainer
from utils.estimates import estimates_function_runtime
from torch.nn.utils.rnn import pack_padded_sequence


class COCOEncoder(nn.Module):
    """
    CNN Model of image captioning
    use pretrained resnet-152
    """

    def __init__(self, embedding_size):
        super(COCOEncoder, self).__init__()

        resnet = resnet152(True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Sequential(
            nn.Linear(resnet.fc.in_features, embedding_size),
            nn.BatchNorm1d(embedding_size),
        )

    def forward(self, x):
        with torch.no_grad():
            output = self.resnet(x)

        output = output.reshape(output.shape[0], -1)
        return self.fc(output)


class COCODecoder(nn.Module):
    """
    Sequential (RNN) part of COCO Caption
    """

    def __init__(self, input_size, hidden_size, vocab_size):
        super(COCODecoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.h2y = nn.Linear(hidden_size, vocab_size)

    def forward(self, feature, captions, lengths):
        embedding = self.embedding(captions)
        embedding = torch.cat((feature.unsqueeze(1), embedding), 1)

        packed = pack_padded_sequence(embedding, lengths, batch_first=True)
        h, c = self.lstm(packed)

        out = self.h2y(h[0])

        return out


class COCOTrainer(ModelTrainer):

    def __init__(self, *args, **kwargs):
        super(COCOTrainer, self).__init__(*args, **kwargs)

    def prepare_model(self, models):
        if self.settings['device'].type == 'cuda':
            self.model = [nn.DataParallel(model) for model in models]
            cudnn.benchmark = True
        else:
            self.model = models

    def update_optimizer(self, outputs, labels):
        encoder, decoder = self.model
        loss = self.calculate_loss(outputs, labels)

        encoder.zero_grad()
        decoder.zero_grad()

        loss.backward()

        self.params['optimizer'].step()

        return loss

    def calculate_loss(self, outputs, labels):
        return self.params['loss_function'](outputs, labels)

    @estimates_function_runtime
    def _train_or_test_single_epoch(self, last_iter, mode):
        if mode not in ('train', 'test'):
            raise ValueError('Invalid Mode')

        is_train = (mode == 'train')
        encoder, decoder = self.model
        if is_train:
            encoder.train()
            decoder.train()
        else:
            encoder.eval()
            decoder.eval()

        epoch_loss = 0
        epoch_size = 0
        epoch_perplexity = 0.

        verbose = self.settings['verbose']
        print_every = self.settings['print_every']
        device = self.settings['device']

        for i, (images, captions, lengths) in enumerate(
                self.data_loader[mode], last_iter + 1):
            real_batch_size = images.shape[0]

            images = images.to(device)
            captions = captions.to(device)
            lengths = lengths.to(device)

            targets = pack_padded_sequence(
                captions, lengths, batch_first=True)[0]

            features = encoder(images)
            outputs = decoder(features, captions, lengths)

            if is_train:
                loss = self.update_optimizer(outputs, targets)
            else:
                loss = self.calculate_loss(outputs, targets)

            perplexity = torch.exp(loss)

            epoch_loss += (loss.item() * real_batch_size)
            epoch_size += real_batch_size
            epoch_perplexity += (perplexity.item() * real_batch_size)

            if is_train and verbose and i % print_every == 0:
                print('Iter: %d, loss = %f, perplexity = %f' %
                      (i, loss.item(), perplexity.item()))

            if self.files is not None:
                self.files['%s_iter' % mode].write(
                    '%d %f %f\n' % (i, loss.item(), perplexity.item()))

        return epoch_loss / epoch_size, epoch_perplexity / epoch_size, i
