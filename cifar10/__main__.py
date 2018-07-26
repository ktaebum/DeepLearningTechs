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

from preprocess.args import get_args
from preprocess.data import prepare_cifar10

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
    loader = prepare_cifar10()
    last_epoch = 0

    # model = GoogleNet(mode='improved', aux=False).to(device)
    model = ResNet(layer_num='50').to(device)
    model_name = model.__class__.__name__ + '_' + model.mode

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=5e-4)

    if pretrained is not None:
        print('load %s...' % pretrained)

        checkpoint = torch.load(os.path.join('./saved_models', pretrained))
        pattern = r'_[0-9]+\.'
        last_epoch = int(re.findall(pattern, pretrained)[-1][1:-1])
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
