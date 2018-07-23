import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader

import torchvision.datasets as datasets
import torchvision.transforms as transforms

import os
import re
import argparse

from models.googlenet import GoogleNet
from utils.model_io import load_parallel_state_dict
from utils.estimates import estimates_function_runtime

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
    '-l',
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

print('=' * 30)
print('Hyperparameter settings')
print('\tBatch Size = %d' % batch_size)
print('\tLearning Rate = %f' % learning_rate)
print('\tNum Epochs = %d' % num_epochs)
print('\tPrint Every = %d' % print_every)
print('\tSave Every = %d' % save_frequency)
print('\tPretrained Model = {}'.format(pretrained))
print('\tVerbose? {}'.format(verbose))
print('\tTraining? {}'.format(is_train))
print('=' * 30)


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
    best_acc = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = GoogleNet(mode='improved').to(device)

    model_name = model.__class__.__name__ + '_' + model.mode
    print('Train with %s' % model_name)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=5e-4)

    if is_train:
        if not os.path.exists('./log'):
            os.mkdir('./log')
        log_file = open(os.path.join('./log', 'train_log.txt'), 'w')

    if pretrained is not None:
        print('load %s...' % pretrained)

        checkpoint = torch.load(os.path.join('./saved_models', pretrained))
        pattern = r'_[0-9]+\.'
        last_epoch = int(re.findall(pattern, pretrained)[-1][1:-1])
        best_acc = checkpoint['acc']
        # model.load_state_dict(checkpoint['state_dict'])
        if device.type == 'cuda':
            load_parallel_state_dict(model, checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        print('loading pretrained model finished')

    if device.type == 'cuda':
        # parallel process
        print('Using parallel processing...')
        model = nn.DataParallel(model)
        cudnn.benchmark = True

    @estimates_function_runtime
    def test(model):
        # eval for one epoch
        model.eval()
        epoch_loss = 0
        epoch_correct = 0
        epoch_size = 0
        for images, labels in loader['test']:
            images = images.to(device)
            labels = labels.to(device)

            batch_size = images.size(0)

            outputs, _, _ = model(images)
            logits = F.log_softmax(outputs, dim=1)
            _, predicts = torch.max(logits, 1)
            corrects = torch.sum(predicts == labels)

            loss = criterion(logits, labels)

            epoch_loss += (loss.item() * batch_size)
            epoch_size += batch_size
            epoch_correct += corrects.item()

        return epoch_loss / epoch_size, epoch_correct / epoch_size

    @estimates_function_runtime
    def train(model, criterion, optimizer, last_iter):
        # train for one epoch
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_size = 0
        for i, (images, labels) in enumerate(loader['train'], last_iter):
            images = images.to(device)
            labels = labels.to(device)

            batch_size = images.size(0)

            outputs, aux1, aux2 = model(images)
            logits = F.log_softmax(outputs, dim=1)
            aux1_logits = F.log_softmax(aux1, dim=1)
            aux2_logits = F.log_softmax(aux2, dim=1)
            _, predicts = torch.max(logits, 1)
            corrects = torch.sum(predicts == labels)
            accuracy = corrects.double() / batch_size

            optimizer.zero_grad()
            loss = criterion(logits, labels)
            aux_loss1 = criterion(aux1_logits, labels)
            aux_loss2 = criterion(aux2_logits, labels)

            loss += (0.3 * (aux_loss1 + aux_loss2))

            loss.backward()
            optimizer.step()

            epoch_loss += (loss.item() * batch_size)
            epoch_size += batch_size
            epoch_correct += corrects.item()

            log_file.write('%d %f %f\n' % (i, loss, accuracy))

            if verbose and i % print_every == 0:
                print('At iter: %d, loss = %f, accuracy = %f' % (i, loss,
                                                                 accuracy))
        return epoch_loss / epoch_size, epoch_correct / epoch_size, i

    if is_train:
        last_iter = 0
        total_train_time = 0
        for epoch in range(last_epoch + 1, last_epoch + 1 + num_epochs):
            elapsed_train_time, (train_loss, train_acc, last_iter) = train(
                model, criterion, optimizer, last_iter)
            elapsed_test_time, (test_loss, test_acc) = test(model)

            print(
                'Train epoch: %d [%f sec elapsed], loss = %f, accuracy = %f'
                % (epoch, elapsed_train_time, train_loss, train_acc))
            print('Test epoch: %d [%f sec elapsed], loss = %f, accuracy = %f'
                  % (epoch, elapsed_test_time, test_loss, test_acc))

            total_train_time += elapsed_train_time

            if test_acc > best_acc:
                best_acc = test_acc

            if save_frequency != 0 and epoch % save_frequency == 0:
                if not os.path.exists('./saved_models'):
                    os.mkdir('./saved_models')

                state = {
                    'state_dict': model.state_dict(),
                    'acc': test_acc,
                    'optimizer': optimizer.state_dict()
                }
                torch.save(
                    state,
                    os.path.join('./saved_models',
                                 '%s_%03d.pth.tar' % (model_name, epoch)))

        print('Training Finished!')
        print('Total train time = %f seconds' % total_train_time)
        print('Best accuracy = %f' % best_acc)
        log_file.close()

    else:
        # eval mode
        _, (test_loss, test_acc) = test(model)
        print('After eval: loss = %f, accuracy = %f' % (test_loss, test_acc))


if __name__ == "__main__":
    main()
