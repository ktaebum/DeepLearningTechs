import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def save_coco_sample(items, filename, nrows=4, ncols=4):

    if len(items) != nrows * ncols:
        raise ValueError(
            'Number of items should be equal to %d' % (nrows * ncols))
    fig, axes = plt.subplots(figsize=(16, 9), nrows=nrows, ncols=ncols)

    i = 0
    for row in range(nrows):
        for col in range(ncols):
            image, caption = items[i]
            image = image.permute(1, 2, 0)
            caption = caption[0]
            axes[row][col].imshow(image)
            axes[row][col].set_title(caption)

            i += 1

    plt.tight_layout()
    plt.savefig(filename)
    pass


def plot_train_log(filename, model_name=None):
    data = np.loadtxt(filename)
    iteration = data[:, 0]
    losses = data[:, 1]
    accuracies = data[:, 2]

    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    axes[0].plot(iteration, losses)

    axes[1].plot(iteration, accuracies)
