import torchvision.datasets as datasets
import torchvision.transforms as transforms

import torch.utils.data.dataloader as dataloader


def prepare_cifar10(batch_size=64):
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
    return __build_loader(data, batch_size)


def prepare_mnist(batch_size=64, resize=32):
    """
    Resize is for lenet
    """
    normalize = transforms.Normalize((0.5,), (1.0,))

    preprocess = {
        'train':
            transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor(), normalize
            ]),
        'test':
            transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor(), normalize
            ])
    }

    data = {
        'train':
            datasets.MNIST(
                './data/',
                train=True,
                download=True,
                transform=preprocess['train']),
        'test':
            datasets.MNIST(
                './data/',
                train=False,
                download=False,
                transform=preprocess['test'])
    }
    return __build_loader(data, batch_size)


def __build_loader(data, batch_size=64):
    loader = {}
    loader['train'] = dataloader.DataLoader(
        data['train'], batch_size=batch_size, shuffle=True)

    loader['test'] = dataloader.DataLoader(
        data['test'], batch_size=batch_size)
