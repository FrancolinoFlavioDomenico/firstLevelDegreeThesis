import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os


import globalVariable as gv
from src import logger


class Plotter:

    def __init__(self, dataset_name, poisoning):
        self.poisoning = poisoning
        self.dataset_name = dataset_name

        self.fig, (self.accuracy_chart, self.loss_chart) = plt.subplots(nrows=2, ncols=1, sharex=True)
        plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.2f'))
        self.accuracy_chart.grid(True)
        self.loss_chart.grid(True)

        self.x_axis = np.arange(gv.ROUNDS_NUM + 1)
        self.loss_chart.set_xlabel('round')

        self.accuracy_chart.set_title('Accuracy over round')
        self.accuracy_chart.set_ylabel("Accuracy")

        self.loss_chart.set_title('Loss over round')
        self.loss_chart.set_ylabel("Loss")
        
        self.accuracy_data = []
        self.loss_data = []

    def plot(self):
        self.accuracy_chart.set_ylabel(f"{self.accuracy_chart.get_ylabel()}\n(peak: {round(self.accuracy_data[len(self.accuracy_data) - 1], 2)})")
        self.loss_chart.set_ylabel(f"{self.loss_chart.get_ylabel()}\n(peak: {round(self.loss_data[len(self.loss_data) - 1], 2)})")

        self.accuracy_chart.plot(self.accuracy_data)
        self.loss_chart.plot(self.x_axis, self.loss_data)
        
        output_plot_dir = f"outputPlot/{'whitPoisoning' if self.poisoning else 'whitoutPoisoning'}/"
        print(f'output dir is : {output_plot_dir}')
        logger.debug(f'output dir is : {output_plot_dir}')
        output_plot_dir = os.path.join(output_plot_dir, f'{self.dataset_name}_accuracy_and_loss.png')
        plt.savefig(output_plot_dir)
