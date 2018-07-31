import torch
import torch.nn as nn
import torch.optim as optim

from models.caption import COCODecoder, COCOEncoder, COCOTrainer

from preprocess.args import get_args
from preprocess.data import prepare_coco_detection
from preprocess.nlp import build_coco_vocabulary

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
    vocabulary = build_coco_vocabulary()
    if verbose:
        print('Building Vocabulary finished, vocabulary length = %d' %
              len(vocabulary))
    loader = prepare_coco_detection(vocabulary, batch_size)

    last_epoch = 0

    encoder = COCOEncoder(256).to(device)
    decoder = COCODecoder(256, 512, len(vocabulary)).to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    params = list(decoder.parameters()) + list(encoder.fc.parameters())
    optimizer = optim.Adam(params, lr=learning_rate, weight_decay=5e-4)

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

    trainer = COCOTrainer(
        (encoder, decoder),
        loader,
        hyperparameters,
        settings,
    )

    if is_train:
        trainer.train()
    else:
        trainer.test()


if __name__ == "__main__":
    main()
