import pickle
import os

from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.models import resnet50
import torch
from typing import Tuple, Dict
# from torch.utils.data import random_split
# from torch.nn import Sequential, Dropout, Linear, Module,CrossEntropyLoss
# from torch import no_grad, max
# import torch
# from torch.cuda import is_available


import numpy as np
import logging

from torchvision import datasets

from torch.utils.data.dataloader import DataLoader
import gc 



class Utils:
    CLIENTS_NUM = 10
    POISONERS_CLIENTS_CID = np.random.randint(0, CLIENTS_NUM, round((CLIENTS_NUM * 30) / 100))
    DATASET_PATH = 'data/torchDownload'

    def __init__(self, dataset_name, classes_number, kernel_size, input_shape, poisoning=False, blockchain=False):
        self.dataset_name = dataset_name
        self.classes_number = classes_number
        self.kernel_size = kernel_size
        self.input_shape = input_shape
        self.poisoning = poisoning
        self.blockchain = blockchain

        self.train_data = self.download_data(True)
        self.test_data = self.download_data(False)
        
        dataset_partition_dir = f"data/partitions/{self.dataset_name}"
        if not os.path.exists(dataset_partition_dir):
            os.makedirs(dataset_partition_dir)

        file = open(os.path.join(dataset_partition_dir, f"test_data.pickle"), "wb")
        try:
            pickle.dump(self.test_data,file)
        finally:
            file.close()

        self.generate_dataset_client_partition()
        
        del(self.train_data)
        del(self.test_data)
        gc.collect()


    def download_data(self, train):
  
        
        if self.dataset_name == 'cifar100':
            data = datasets.CIFAR100(
                root=Utils.DATASET_PATH,
                train=train,
                download=True
                # transform=train_transform if train else test_transform
            )
        elif self.dataset_name == 'cifar10':
            data = datasets.CIFAR10(
                root=Utils.DATASET_PATH,
                train=train,
                download=True
                #                 transform=train_transform if train else test_transform
            )
        else:
            data = datasets.MNIST(
                root=Utils.DATASET_PATH,
                train=train,
                download=True
                #                 transform=train_transform if train else test_transform
            )
            # serve?
            # self.train_data.data = self.train_data.data.reshape(self.train_data.data.shape[0],self.train_data.data.shape[1], self.train_data.data.shape[2],1)

        if not train:
            stats = ((0.5), (0.5)) if self.dataset_name == 'mnist' else ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            test_transform = Compose([
                ToTensor(),
                Normalize(*stats)
            ])
            data.transform = test_transform

        return data

    def generate_dataset_client_partition(self):
        partition_lenght = np.full(Utils.CLIENTS_NUM, len(self.train_data.data) / Utils.CLIENTS_NUM).astype(
            int).tolist()
        partitions = torch.utils.data.random_split(self.train_data, partition_lenght)

        dataset_partition_dir = f"data/partitions/{self.dataset_name}"
        if not os.path.exists(dataset_partition_dir):
            os.makedirs(dataset_partition_dir)

        for i, partition in enumerate(partitions):
            file = open(os.path.join(dataset_partition_dir, f"partition_{i}.pickle"), "wb")
            try:
                pickle.dump(partition,file)
            finally:
                file.close()  
            
    def test(
        self,
        model
    ) -> Tuple[float, float]:
        """Validate the network on the entire test set."""
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        testloader = DataLoader(self.get_test_data())
        
        criterion = torch.nn.CrossEntropyLoss()
        correct = 0
        total = 0
        loss = 0.0
        model.to(device)
        with torch.no_grad():
            model.eval()
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        model.to('cpu')
        return loss, accuracy
            
    def get_test_data(self):
            
        file = open(os.path.join(f"data/partitions/{self.dataset_name}",
                               f"test_data.pickle"), "rb")
        try:
            test_data = pickle.load(file)
        finally:
            file.close()
  
        return test_data      

    def get_model(self) -> torch.nn.Module:
        model = resnet50(weights='IMAGENET1K_V1')
            
        num_ftrs = model.fc.in_features

        #test
        model.fc = torch.nn.Linear(num_ftrs, 256)
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(num_ftrs, self.classes_number)
        )
        
        return model
    
        #initial arch
        # model.fc = torch.nn.Linear(num_ftrs, 1024)
        # model.fc = torch.nn.Sequential(
        #     torch.nn.Dropout(0.5),
        #     torch.nn.Linear(num_ftrs, 1024),
        #     torch.nn.Dropout(0.2),
        #     torch.nn.Linear(1024, 512),
        #     torch.nn.Dropout(0.2),
        #     torch.nn.Linear(512, 256),
        #     torch.nn.Dropout(0.2),
        #     torch.nn.Linear(256, 128),
        #     torch.nn.Dropout(0.2),
        #     torch.nn.Linear(128, self.classes_number)
        # )
        
        # return model

    @classmethod
    def printLog(cls, msg):
        print(msg)
        logging.log(logging.INFO, msg)
