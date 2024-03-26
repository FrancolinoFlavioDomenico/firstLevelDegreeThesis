import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
import os

import globalVariable as gv
from src import logger
import datasetLabelMapping

class Plotter:

    def __init__(self, dataset_name, poisoning):
        self.poisoning = poisoning
        self.dataset_name = dataset_name

    def line_chart_plot(self, accuracy_data, loss_data):
        line_chart_fig, (accuracy_chart, loss_chart) = plt.subplots(nrows=2, ncols=1, sharex=True)
        accuracy_chart.grid(True)
        loss_chart.grid(True)
        accuracy_chart.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
        loss_chart.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.2f}'))

        x_axis = np.arange(gv.ROUNDS_NUM + 1)
        loss_chart.set_xlabel('round')

        accuracy_chart.set_title('Accuracy over round')
        accuracy_chart.set_ylabel(f"Accuracy\n(peak: {round(accuracy_data[len(accuracy_data) - 1], 2)})")

        loss_chart.set_title('Loss over round')
        loss_chart.set_ylabel(f"Loss\n(peak: {round(loss_data[len(loss_data) - 1], 2)})")

        accuracy_chart.plot(accuracy_data)
        loss_chart.plot(x_axis, loss_data)
        self.set_save_fig_path('accuracy_and_loss')

    def confusion_matrix_chart_plot(self, data):
        confusion_matrix_fig, confusion_matrix_chart = plt.subplots()
        confusion_matrix_fig.set_figwidth(10)
        confusion_matrix_fig.set_figheight(10)
        if 'cifar100' in self.dataset_name:
            confusion_matrix_fig.set_figwidth(50)
            confusion_matrix_fig.set_figheight(50)
        confusion_matrix_chart.set_xlabel("correct class")
        confusion_matrix_chart.set_ylabel("predicted class")
        sns.heatmap(data, annot=True, cmap="Blues_r", linewidths=2, square=True)
        if 'cifar10' == self.dataset_name:
            confusion_matrix_chart.xaxis.set_ticklabels(datasetLabelMapping.cifar10Label)
            confusion_matrix_chart.yaxis.set_ticklabels(datasetLabelMapping.cifar10Label)
        if 'cifar100' == self.dataset_name:
            confusion_matrix_chart.xaxis.set_ticklabels(datasetLabelMapping.cifar100FineLabel)
            confusion_matrix_chart.yaxis.set_ticklabels(datasetLabelMapping.cifar100FineLabel)
        self.set_save_fig_path('confusionMatrix')

    def set_save_fig_path(self, chart_name):
        output_plot_dir = f"outputPlot/{'whitPoisoning' if self.poisoning else 'whitoutPoisoning'}/"
        print(f'output dir is : {output_plot_dir}')
        logger.debug(f'output dir is : {output_plot_dir}')
        output_plot_dir = os.path.join(output_plot_dir,
                                       f"{self.dataset_name}_{chart_name}_{'poisoned' if self.poisoning else ''}.png")
        plt.savefig(output_plot_dir)
