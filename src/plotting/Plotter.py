import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
import os

from  src.utils.globalVariable import cifar10Label,cifar100FineLabel

# from src.utils.Utils import Utils
warnings.filterwarnings('ignore')

class Plotter:

    poisoning = None
    blockchain = None
    dataset_name = None
    rounds_number = None
    
    @classmethod
    def configure_plotter(cls, dataset_name,rounds_number, poisoning, blockchain):
        # Utils.printLog("plotter initialization")
        cls.poisoning = poisoning
        cls.blockchain = blockchain
        cls.dataset_name = dataset_name
        cls.rounds_number = rounds_number

    ########################################################################################
    # build a line chart accuracy chart
    ########################################################################################
    @classmethod
    def line_chart_plot(cls, accuracy_data, loss_data):
        line_chart_fig, (accuracy_chart, loss_chart) = plt.subplots(nrows=2, ncols=1, sharex=True)
        accuracy_chart.grid(True)
        loss_chart.grid(True)
        accuracy_chart.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
        loss_chart.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.2f}'))

        x_axis = np.arange(cls.rounds_number + 1)
        loss_chart.set_xlabel('round')

        accuracy_chart.set_title(f'{cls.dataset_name} Accuracy over round {"poisoned" if cls.poisoning else ""} {"blockchain" if cls.blockchain else "noBlockchain"}')
        accuracy_chart.set_ylabel(f"Accuracy\n(peak: {round(accuracy_data[len(accuracy_data) - 1], 2)})")

        loss_chart.set_title(f'{cls.dataset_name} Loss over round {"poisoned" if cls.poisoning else ""}')
        loss_chart.set_ylabel(f"Loss\n(peak: {round(loss_data[len(loss_data) - 1], 2)})")

        accuracy_chart.plot(accuracy_data)
        loss_chart.plot(x_axis, loss_data)
        cls.save_chart('accuracy_and_loss')

    ########################################################################################
    # build a confusion matrixx chart
    ########################################################################################
    @classmethod
    def confusion_matrix_chart_plot(cls, data):
        confusion_matrix_fig, confusion_matrix_chart = plt.subplots()
        confusion_matrix_fig.set_figwidth(10)
        confusion_matrix_fig.set_figheight(10)
        if 'cifar100' in cls.dataset_name:
            confusion_matrix_fig.set_figwidth(50)
            confusion_matrix_fig.set_figheight(50)
        confusion_matrix_chart.set_title(f'{cls.dataset_name} confusion matrix {"poisoned" if cls.poisoning else ""}  {"blockchain" if cls.blockchain else "noBlockchain"}')
        confusion_matrix_chart.set_xlabel("correct class")
        confusion_matrix_chart.set_ylabel("predicted class")
        sns.heatmap(data, annot=True, cmap="Blues_r", linewidths=2, square=True)
        if 'cifar10' == cls.dataset_name:
            confusion_matrix_chart.xaxis.set_ticklabels(cifar10Label)
            confusion_matrix_chart.yaxis.set_ticklabels(cifar10Label)
        if 'cifar100' == cls.dataset_name:
            confusion_matrix_chart.xaxis.set_ticklabels(cifar100FineLabel)
            confusion_matrix_chart.yaxis.set_ticklabels(cifar100FineLabel)
        cls.save_chart('confusionMatrix')

    ########################################################################################
    # save a chart into local dir
    ########################################################################################
    @classmethod
    def save_chart(cls, chart_name):
        output_plot_dir = f"outputPlot/{'blockchain' if cls.blockchain else 'noBlockchain'}/{'poisoning' if cls.poisoning else 'noPoisoning'}/"
        output_plot_dir = os.path.join(output_plot_dir,
                                       f"{cls.dataset_name}_{chart_name}_{'poisoned' if cls.poisoning else ''}.png")
        plt.savefig(output_plot_dir)

    @classmethod    
    def stacked_bar_chart_plot(cls,clients_num,classes_num,data,dataset_name):
        # Fill missing class with 0 and convert to array
        filled_data = {}
        for client, cls in data.items():
            filled_data[client] = {item: cls.get(item, 0) for item in np.arange(classes_num)}  

        #TODO review gor whowing legend
        fig, ax = plt.subplots()
        width = 0.8
        for client in filled_data.keys():
            bottom = np.zeros(3)
            for cls,cls_counter in list(filled_data[client].items()):
                # label = cls
                # if(dataset_name == 'cifar10'):
                #     label = cifar10Label[label]
                # elif(dataset_name == 'cifar100'):
                #     label = None
                    
                # bar = ax.bar(client,cls_counter,width,label=label,bottom=bottom)
                bar = ax.bar(client,cls_counter,width,bottom=bottom)
                bottom += cls_counter
                if(dataset_name != 'cifar100'):
                    ax.bar_label(bar,label_type='center')

        ax.set_xlabel("clients")
        ax.set_ylabel("classes")
        ax.set_title(f"{dataset_name} Class-client distribution")
        # fig.legend(loc="upper right")
        # fig.set_figwidth(15)
        # fig.set_figheight(15)

        output_plot_dir = f"outputPlot/"
        output_plot_dir = os.path.join(output_plot_dir,f"{dataset_name}_class_client_distribution.png")
        plt.savefig(output_plot_dir)

    
    
    
    
    
    
