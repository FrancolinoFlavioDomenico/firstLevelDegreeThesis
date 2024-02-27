import matplotlib.pyplot as plt
import numpy as np

import os

from src import ROUNDS_NUM


class Plotter:

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

        self.fig, (self.ay_accuracy, self.ay_loss) = plt.subplots(nrows=2, ncols=1, sharex=True)
        self.x_axis = np.arange(ROUNDS_NUM + 1)
        self.ay_loss.set_xlabel('round')

        self.ay_accuracy.set_title('Accuracy over round')
        self.ay_accuracy.set_ylabel("Accuracy")

        self.ay_loss.set_title('Loss over round')
        self.ay_loss.set_ylabel("Loss")

        self.accuracy_data = []  # np.array()
        self.loss_data = []  # np.array()

    def plot(self):
        self.ay_accuracy.plot(self.x_axis, self.accuracy_data)
        self.ay_loss.plot(self.x_axis, self.loss_data)

        output_plot_dir = os.path.join('outputPlot/', f'{self.dataset_name}_accuracy_and_loss_whitout_poisoning.png')
        plt.savefig(output_plot_dir)
