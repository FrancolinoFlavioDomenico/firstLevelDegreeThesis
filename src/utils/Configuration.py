import pickle
import os
from torchvision.transforms import ToTensor, Normalize, Compose
import requests
import torch
import numpy as np
from torchvision import datasets
import gc
from src.utils.globalVariable  import seed_value
from src.plotting.Plotter import Plotter as plt
from src.utils.Utils import Utils
from src.utils.globalVariable import blockchainApiPrefix,blockchainPrivateKeys

torch.manual_seed(seed_value)
np.random.seed(seed_value)


class Configuration:
    CLIENTS_NUM = 10
    POISONERS_CLIENTS_CID = np.random.randint(0, CLIENTS_NUM, round((CLIENTS_NUM * 30) / 100))
    DATASET_PATH = 'data/torchDownload'
    ROUNDS_NUMBER = 5

    def __init__(self, dataset_name, classes_number, poisoning=False, blockchain=False):
        self.dataset_name = dataset_name
        self.classes_number = classes_number
        self.poisoning = poisoning
        self.blockchain = blockchain
        
        plt.configure_plotter(self.dataset_name, Configuration.ROUNDS_NUMBER, self.poisoning,
                                       self.blockchain)
        
        self.set_data()
        
        if self.poisoning:
            Utils.printLog(f"poisoner are {Configuration.POISONERS_CLIENTS_CID}")
            
        if self.blockchain:
            requests.post(f'{blockchainApiPrefix}/configure/training',
                json={'datasetName': self.dataset_name,'datasetClassNumber':self.classes_number,'maxRound':Configuration.ROUNDS_NUMBER,'clientsNum':Configuration.CLIENTS_NUM})
        gc.collect()
        
    ########################################################################################
    # prepare dataset
    ########################################################################################
    def set_data(self):
        dataset_partition_dir = f"data/partitions/{self.dataset_name}"
        if not os.path.exists(dataset_partition_dir):
            os.makedirs(dataset_partition_dir)
            
            train_data = self.download_data(True)
            test_data = self.download_data(False)

            file = open(os.path.join(dataset_partition_dir, f"test_data.pickle"), "wb")
            try:
                pickle.dump(test_data, file)
            finally:
                file.close()

            self.generate_dataset_client_partition(dataset_partition_dir,train_data)

            del (train_data)
            del (test_data)

    ########################################################################################
    # Download and save dataset in local dir
    ########################################################################################
    def download_data(self, train):
        if self.dataset_name == 'cifar100':
            data = datasets.CIFAR100(
                root=Configuration.DATASET_PATH,
                train=train,
                download=True
            )
        elif self.dataset_name == 'cifar10':
            data = datasets.CIFAR10(
                root=Configuration.DATASET_PATH,
                train=train,
                download=True
            )
        else:
            data = datasets.MNIST(
                root=Configuration.DATASET_PATH,
                train=train,
                download=True
            )

        if not train:
            stats = ((0.5), (0.5)) if self.dataset_name == 'mnist' else (
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            test_transform = Compose([
                ToTensor(),
                Normalize(*stats)
            ])
            data.transform = test_transform

        return data

    ########################################################################################
    # Generete  a dataset partition used by single federated client
    ########################################################################################
    def generate_dataset_client_partition(self,dataset_partition_dir,train_data):
        partition_lenght = np.random.multinomial(len(train_data.data), np.random.dirichlet(np.ones(Configuration.CLIENTS_NUM) * 42)).astype(
            int).tolist()
        partitions = torch.utils.data.random_split(train_data, partition_lenght)

        class_client_distribution = {}
        for i, partition in enumerate(partitions):
            class_counts = {}
            for j, (image, label) in enumerate(partition):
                if label not in class_counts:
                    class_counts[label] = 1
                else:
                    class_counts[label] += 1
            class_client_distribution[i] = class_counts
            
            
            file = open(os.path.join(dataset_partition_dir, f"partition_{i}.pickle"), "wb")
            try:
                pickle.dump(partition, file)
            finally:
                file.close()
                
        plt.stacked_bar_chart_plot(Configuration.CLIENTS_NUM,self.classes_number,class_client_distribution,self.dataset_name)
