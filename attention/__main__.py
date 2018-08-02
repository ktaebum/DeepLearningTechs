import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torch.nn.utils.rnn import pack_padded_sequence

from models.attention import AttentionDecoder, AttentionEncoder

from preprocess.args import get_args
from preprocess.data import prepare_coco_detection
from preprocess.nlp import build_coco_vocabulary, Vocabulary

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


def main():
    cudnn.benchmark = True
    vocabulary = build_coco_vocabulary()
    if verbose:
        print('Building Vocabulary finished, vocabulary length = %d' %
              len(vocabulary))
    loader = prepare_coco_detection(vocabulary, batch_size)

    last_epoch = 0

    encoder = AttentionEncoder().to(device)
    decoder = AttentionDecoder(256, 256, len(vocabulary)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        decoder.parameters(), lr=learning_rate, weight_decay=5e-4)

    hyperparameters = {
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'num_epochs': num_epochs,
        'optimizer': optimizer,
        'loss_function': criterion
    }

    settings = {
        'print_every': print_every,
        'verbose': verbose,
        'save_log': is_log,
        'start_epoch': last_epoch + 1,
        'save_model': save_frequency,
        'name': 'coco_caption',
        'device': device
    }

    def train(criterion, optimizer, last):
        for i, (images, captions, lengths) in enumerate(
                loader['train'], last):

            images = images.to(device)
            captions = captions.to(device)

            features = encoder(images)
            output = decoder(features, captions, lengths)

            targets = pack_padded_sequence(
                captions, lengths, batch_first=True)[0].to(device)
            output = pack_padded_sequence(
                output, lengths, batch_first=True)[0].to(device)

            # for memory
            del images
            del captions

            optimizer.zero_grad()

            loss = criterion(output, targets)
            loss.backward()

            optimizer.step()

            if i % print_every == 0:
                print('iter %d: loss %f' % (i, loss))

        return i

    def test():

        with torch.no_grad():
            decoder.eval()
            encoder.eval()

            for i, (images, _, _) in enumerate(loader['test']):

                real_batch_size = images.shape[0]
                images = images.to(device)
                start_input = torch.LongTensor([vocabulary[Vocabulary.SOS]] *
                                               real_batch_size).to(device)
                start_input.reshape(real_batch_size, 1)

                feature = encoder(images)
                sampled_ids = decoder.sample(start_input, feature)

                sampled_ids = sampled_ids[1].cpu().numpy()

                generated_caption = [vocabulary[idx] for idx in sampled_ids]
                print(' '.join(generated_caption))

    if is_train:
        last = 0
        for epoch in range(num_epochs):
            last = train(criterion, optimizer, last)
            test()

    else:
        test()


if __name__ == "__main__":
    main()
