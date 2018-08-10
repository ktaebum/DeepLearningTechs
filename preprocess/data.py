import torch
import torch.utils.data.dataloader as dataloader

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from preprocess.nlp import sentence2words, Vocabulary


def extract_coco_feature(model, batch_size=64):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224, pad_if_needed=True),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data = datasets.CocoCaptions(
        './data/train2014',
        './data/annotations/captions_train2014.json',
        transform=transform)

    loader = dataloader.DataLoader(data, batch_size=batch_size)
    model.eval()

    features = []
    captions = []

    length = len(loader)
    for i, (images, captions) in enumerate(loader):
        if torch.cuda.is_available():
            images = images.cuda()
        feature = model(images)
        features.append(feature.cpu())
        captions.extend(captions[0])

        if i % 100 == 0:
            print('[%d/%d] finished' % (i, length))

    return torch.cat(features, 0), captions


def prepare_coco_from_feature(vocabulary,
                              fea_file,
                              batch_size=64,
                              max_len=50):
    all_data = torch.load(fea_file)
    print(all_data)

    def _map_wrapper(lists, function, casting=None, **kwargs):
        mapping_result = list(map(lambda x: function(x, **kwargs), lists))
        if casting:
            mapping_result = casting(mapping_result)
        return mapping_result

    def collate_coco(data):
        data.sort(
            key=lambda x: len(x[1]), reverse=True)  # sort by caption length

        features, captions = zip(*data)

        def _extract_ith_caption(captions, i=0):
            """
            collect ith-caption among 5 captions
            default first
            """
            return captions[i]

        def _caption2idx(caption):
            words = sentence2words(caption)

            idxs = _map_wrapper([Vocabulary.SOS] + words + [Vocabulary.EOS],
                                vocabulary.__getitem__)

            # check padding
            pad_length = max_len - len(idxs)
            if pad_length > 0:
                # pad
                idxs += [vocabulary[Vocabulary.PAD]] * pad_length

            if len(idxs) > max_len:
                # cutoff
                idxs = idxs[:max_len]

            return idxs

        def _min_length(caption):
            return min(len(caption), max_len)

        # images = torch.stack(images, 0)
        # captions = _map_wrapper(captions, _extract_ith_caption, i=0)
        captions = _map_wrapper(
            captions, _caption2idx, casting=torch.LongTensor)
        lengths = _map_wrapper(captions, _min_length)

        return features, captions, lengths

    return dataloader.DataLoader(
        all_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_coco),


def prepare_coco_detection(vocabulary, batch_size=64, deprecated=30):
    """
    vocabulary: coco vocabulary
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224, pad_if_needed=True),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data = {
        'train':
            datasets.CocoCaptions(
                './data/train2014',
                './data/annotations/captions_train2014.json',
                transform=transform),
        'test':
            datasets.CocoCaptions(
                './data/val2014',
                './data/annotations/captions_val2014.json',
                transform=transform,
            ),
    }

    def _map_wrapper(lists, function, casting=None, **kwargs):
        mapping_result = list(map(lambda x: function(x, **kwargs), lists))
        if casting:
            mapping_result = casting(mapping_result)
        return mapping_result

    def collate_coco(data):

        data.sort(
            key=lambda x: len(sentence2words(x[1][0])),
            reverse=True)  # sort by caption length

        def _extract_ith_caption(captions, i=0):
            """
            collect ith-caption among 5 captions
            default first
            """
            return sentence2words(captions[i])

        images, captions = zip(*data)
        captions = _map_wrapper(captions, _extract_ith_caption, i=0)
        lengths = _map_wrapper(captions, len)
        max_len = max(lengths)

        def _caption2idx(caption):
            # words = sentence2words(caption)
            words = caption

            idxs = _map_wrapper([Vocabulary.SOS] + words + [Vocabulary.EOS],
                                vocabulary.__getitem__)

            # check padding
            pad_length = max_len - len(idxs)
            if pad_length > 0:
                # pad
                idxs += [vocabulary[Vocabulary.PAD]] * pad_length

            if len(idxs) > max_len:
                # cutoff
                idxs = idxs[:max_len]

            return idxs

        images = torch.stack(images, 0)
        captions = _map_wrapper(
            captions, _caption2idx, casting=torch.LongTensor)

        return images, captions, lengths

    return {
        'train':
            dataloader.DataLoader(
                data['train'],
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_coco),
        'test':
            dataloader.DataLoader(
                data['test'], batch_size=batch_size, collate_fn=collate_coco)
    }


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

    return loader
