import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torch.nn.utils.rnn import pack_padded_sequence

from models.attention import AttentionDecoder, AttentionEncoder

from preprocess.args import get_args
from preprocess.data import prepare_coco_detection, extract_coco_feature, prepare_coco_from_feature
from preprocess.nlp import build_coco_vocabulary

from utils.model_io import load_parallel_state_dict

args = get_args()

# Hyperparameters
batch_size = args.batch_size
learning_rate = args.learning_rate
num_epochs = args.num_epochs
print_every = args.print_every
save_frequency = args.save_every
pretrained = args.model
verbose = args.verbose
is_train = args.train
is_log = args.log
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
feature_data = 'train_feature.pth'


def main():
    cudnn.benchmark = True
    vocabulary = build_coco_vocabulary()
    if verbose:
        print('Building Vocabulary finished, vocabulary length = %d' %
              len(vocabulary))

    encoder = AttentionEncoder().to(device)
    decoder = AttentionDecoder(512, 512, len(vocabulary)).to(device)
    """
    load_parallel_state_dict(
        decoder,
        torch.load('./checkpoints/attention_000.pth.tar')['decoder'])
    """

    encoder = nn.DataParallel(encoder)
    decoder = nn.DataParallel(decoder)

    loader = prepare_coco_detection(vocabulary, batch_size)['train']
    if feature_data is None:
        feas, caps = extract_coco_feature(encoder, 64)
        torch.save((feas, caps), 'train_feature.pth')

        exit(0)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(
        decoder.parameters(), lr=learning_rate, weight_decay=5e-4)

    def train(criterion, optimizer, last):
        encoder.train()
        decoder.train()
        for i, (features, captions, lengths) in enumerate(loader, last + 1):

            # images = images.to(device)
            captions = captions.to(device)

            lengths = list(map(lambda x: x - 1, lengths))

            # features = encoder(images)
            features = features.to(device)
            features = encoder(features)
            output = decoder(features, captions[:, :-1], lengths)
            output = pack_padded_sequence(
                output, lengths, batch_first=True)[0]

            captions = pack_padded_sequence(
                captions[:, 1:], lengths, batch_first=True)[0]

            # update
            optimizer.zero_grad()
            loss = criterion(output, captions)
            loss.backward()
            optimizer.step()

            if i % print_every == 0:
                print('iter %d: loss = %f' % (i, loss.item()))

        return i

    if is_train:
        last = -1
        for epoch in range(num_epochs):
            last = train(criterion, optimizer, last)

            state = {
                'decoder': decoder.state_dict(),
            }

            torch.save(state,
                       './checkpoints/attention_%03d.pth.tar' % (epoch))
            print('epoch %d finished' % epoch)


if __name__ == "__main__":
    main()
