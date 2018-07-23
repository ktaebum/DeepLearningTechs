import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def plot_train_log(filename, model_name=None):
    data = np.loadtxt(filename)
    iteration = data[:, 0]
    losses = data[:, 1]
    accuracies = data[:, 2]

    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    axes[0].plot(iteration, losses)

    axes[1].plot(iteration, accuracies)
