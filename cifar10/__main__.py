import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.dataloader as dataloader

import torchvision.datasets as datasets
import torchvision.transforms as transforms

import os
import re
import argparse

from models.googlenet import GoogleNet, GoogleNetTrainer
from models.resnet import ResNet, ResNetTrainer
from utils.model_io import load_parallel_state_dict

parser = argparse.ArgumentParser()
parser.add_argument(
    '-p',
    '--print-every',
    help='print every specific number of iteration',
    metavar='',
    type=int,
    default=100)
parser.add_argument(
    '-b',
    '--batch-size',
    help='set number of batch size',
    metavar='',
    type=int,
    default=128)
parser.add_argument(
    '-e',
    '--num-epochs',
    help='set number of total epochs',
    metavar='',
    type=int,
    default=5)
parser.add_argument(
    '-lr',
    '--learning-rate',
    help='set training learning rate',
    metavar='',
    type=float,
    default=1e-3)
parser.add_argument(
    '-v',
    '--verbose',
    help='set whether verbose train or not',
    action='store_true')
parser.add_argument(
    '-t',
    '--train',
    help='set whether run in train mode or not',
    action='store_true')
parser.add_argument(
    '-l',
    '--log',
    help='set whether save some result files',
    action='store_true')
parser.add_argument(
    '-s',
    '--save-every',
    help='if set, save model while training',
    metavar='',
    type=int,
    default=0)
parser.add_argument(
    '-m',
    '--model',
    help='load_pretrained model',
    metavar='',
    type=str,
    default=None)
args = parser.parse_args()

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


def prepare_data():
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.247, 0.243, 0.261))
    preprocess = {
        'train':
            transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(), normalize
            ]),
        'test':
            transforms.Compose([transforms.ToTensor(), normalize])
    }

    data = {
        'train':
            datasets.CIFAR10(
                './data/',
                train=True,
                download=True,
                transform=preprocess['train']),
        'test':
            datasets.CIFAR10(
                './data/',
                train=False,
                download=False,
                transform=preprocess['test'])
    }

    loader = {
        'train':
            dataloader.DataLoader(
                data['train'], batch_size=batch_size, shuffle=True),
        'test':
            dataloader.DataLoader(data['test'], batch_size=batch_size)
    }
    return loader


def main():
    loader = prepare_data()
    last_epoch = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model = GoogleNet(mode='improved', aux=False).to(device)
    model = ResNet(layer_num='152').to(device)
    model_name = model.__class__.__name__ + '_' + model.mode

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=5e-4)

    if pretrained is not None:
        print('load %s...' % pretrained)

        checkpoint = torch.load(os.path.join('./saved_models', pretrained))
        pattern = r'_[0-9]+\.'
        last_epoch = int(re.findall(pattern, pretrained)[-1][1:-1])
        # model.load_state_dict(checkpoint['state_dict'])
        if device.type == 'cuda':
            load_parallel_state_dict(model, checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        print('loading pretrained model finished')
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
        'name': model_name,
        'device': device
    }

    trainer = ResNetTrainer(model, loader, hyperparameters, settings)
    # trainer = GoogleNetTrainer(model, loader, hyperparameters, settings)
    if is_train:
        trainer.train()
    else:
        trainer.test()


if __name__ == "__main__":
    main()
