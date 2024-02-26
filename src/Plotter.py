import matplotlib.pyplot as plt
import numpy as np

import Model

import os


class Plotter:

    def __init__(self):
        self.fig, (self.ay_accuracy, self.ay_loss) = plt.subplots(nrows=2, ncols=1, sharex=True)
        self.x_axis = np.arange(Model.Model.ROUNDS_NUM + 1)
        self.ay_loss.set_xlabel('Epochs')

        self.ay_accuracy.set_title('Accuracy over epochs')
        self.ay_accuracy.set_ylabel("Accuracy")

        self.ay_loss.set_title('Loss over epochs')
        self.ay_loss.set_ylabel("Loss")

        self.accuracy_data = []  # np.array()
        self.loss_data = []  # np.array()

    def plot(self):
        self.ay_accuracy.plot(self.x_axis, self.accuracy_data)
        self.ay_loss.plot(self.x_axis, self.loss_data)

        output_plot_dir = os.path.join('outputPlot/', 'accuracy_and_loss_whitout_poisoning.png')
        plt.savefig(output_plot_dir)
