import torch
import torch.nn as nn
import torch.optim as optim

import os

from models.simpleconv import SimpleConv

from utils.model_io import load_parallel_state_dict

from preprocess.args import get_args
from preprocess.data import prepare_cifar10


def main(args):
    loader = prepare_cifar10(args.batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SimpleConv().to(device)
    model_name = model.__class__.__name__

    loss_function = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=5e-4)

    last_epoch = 0
    best_accuracy = 0
    if args.model is not None:
        checkpoint = torch.load(os.path.join('./checkpoints', args.model))
        load_parallel_state_dict(model, checkpoint['state_dict'])
        last_epoch = checkpoint['epoch']
        best_accuracy = checkpoint['acc']

    def train(optimizer, loss_function, last_iter=-1):
        model.train()
        epoch_loss = 0
        epoch_corrects = 0
        epoch_size = 0
        for i, (images, labels) in enumerate(loader['train'], last_iter):
            batch_size = images.shape[0]

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predicted = torch.max(outputs, 1)[1]
            corrects = torch.sum(predicted == labels)

            optimizer.zero_grad()
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            accuracy = corrects.item() / batch_size

            epoch_loss += (loss.item() * batch_size)
            epoch_corrects += corrects.item()
            epoch_size += batch_size

            if i % args.print_every == 0:
                print('Iter %d: loss = %f, accuracy = %f' % (i, loss,
                                                             accuracy))

        loss = epoch_loss / epoch_size
        accuracy = epoch_corrects / epoch_size

        return loss, accuracy, i

    def test(loss_function, last_iter=-1):
        model.eval()
        epoch_loss = 0
        epoch_corrects = 0
        epoch_size = 0
        for i, (images, labels) in enumerate(loader['test'], last_iter):
            with torch.no_grad():
                batch_size = images.shape[0]

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                predicted = torch.max(outputs, 1)[1]
                corrects = torch.sum(predicted == labels)

                loss = loss_function(outputs, labels)

                accuracy = corrects.item() / batch_size

                epoch_loss += (loss.item() * batch_size)
                epoch_corrects += corrects.item()
                epoch_size += batch_size

        loss = epoch_loss / epoch_size
        accuracy = epoch_corrects / epoch_size

        return loss, accuracy, i

    last_t = -1
    last_e = -1
    for epoch in range(last_epoch + 1, last_epoch + 1 + args.num_epochs):
        t_l, t_a, last_t = train(optimizer, loss_function, last_t)

        print('Epoch [%d / %d]: train loss = %f, trian accuracy = %f' %
              (epoch, args.num_epochs, t_l, t_a))

        e_l, e_a, last_e = test(loss_function, last_e)

        print('Epoch [%d / %d]: test loss = %f, test accuracy = %f' %
              (epoch, args.num_epochs, e_l, e_a))

        if not os.path.exists('./checkpoints'):
            os.mkdir('./checkpoints')

        state = {
            'state_dict': model.state_dict(),
            'epoch': epoch,
            'acc': e_a,
        }

        torch.save(
            state,
            os.path.join('./checkpoints',
                         '%s_%03d.pth.tar' % (model_name, epoch)))


if __name__ == "__main__":
    args = get_args()
    main(args)
