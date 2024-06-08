import pickle
import os

import ray.data
import torch
from torchvision.transforms import ToTensor, Normalize, Compose, Resize
from torchvision.transforms import RandomHorizontalFlip, RandomCrop
from torchvision.models import efficientnet_b0, resnet50, resnet18
import warnings


import numpy as np
import logging

from torchvision import datasets

import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid
import multiprocessing as mp
import gc 
import ray



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

        # print("------------------------------------------------------------------------------")
        # self.get_model()
        # print("------------------------------------------------------------------------------")

        # test code.....print image 
        # train_data.transform = Utils.train_transform
        # batch_size = 128
        # train_dl = DataLoader(self.train_data, batch_size, num_workers=0, pin_memory=True, shuffle=True)
        # for batch in train_dl:
        #     images, labels = batch
        #
        #     fig, ax = plt.subplots(figsize=(7.5, 7.5))
        #     ax.set_yticks([])
        #     ax.set_xticks([])
        #     ax.imshow(make_grid(images[:20], nrow=5).permute(1, 2, 0))
        #     break
        # plt.show()


        self.generate_dataset_client_partition()
        
        del(self.train_data)
        del(self.test_data)
        gc.collect()


    def download_data(self, train):
        stats = ((0.5), (0.5)) if self.dataset_name == 'mnist' else ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        test_transform = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(*stats)
        ])
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
            # with open(os.path.join(dataset_partition_dir, f"partition_{i}.pickle"), "wb") as f:
            #     pickle.dump(partition, f)  
            
            
    def get_test_data_loader(self,batch_size,datal_loader=True):
        # with open(os.path.join(f"data/partitions/{self.utils.dataset_name}",
        #                        f"test_data.pickle"), "rb") as f:
        #     test_data = pickle.load(f)
            
        file = open(os.path.join(f"data/partitions/{self.dataset_name}",
                               f"test_data.pickle"), "rb")
        try:
            test_data = pickle.load(file)
        finally:
            file.close()
            
        if datal_loader:
            return  DataLoader(test_data, batch_size=batch_size, shuffle=False)    
        return test_data      

    def get_model(self) -> torch.nn.Module:
        """Loads EfficienNetB0 from TorchVision."""
        efficientnet = efficientnet_b0(weights='IMAGENET1K_V1')
        # last_conv_layer_found = False

        # for name, layer in efficientnet.named_children():
        #     if isinstance(layer, torch.nn.Conv2d):  # Identifica i livelli convoluzionali
        #         # Se abbiamo trovato l'ultimo livello convoluzionale, smetti di congelare
        #         if last_conv_layer_found:
        #             break
        #         # Congela i parametri del livello convoluzionale
        #         for param in layer.parameters():
        #             param.requires_grad = False
        #         # Se questo è l'ultimo livello convoluzionale, impostalo come trovato
        #         last_conv_layer_found = True
        #     else:
        #         # Se il livello non è convoluzionale, congela i suoi parametri
        #         if not last_conv_layer_found:
        #             for param in layer.parameters():
        #                 param.requires_grad = False

        # PROBABILE BUONO
        # for param in efficientnet.parameters():
        #     param.requires_grad = False
        # unfreeze_layers = efficientnet.features[-3:]
        # for feature in unfreeze_layers:
        #     for param in feature.parameters():
        #         param.requires_grad = True

        for name, param in efficientnet.named_parameters():
            param.requires_grad = False
        
        efficientnet = torch.nn.Sequential(efficientnet, torch.nn.Dropout(p=0.4, inplace=False), torch.nn.Linear(1000,self.classes_number))

        #self made buono? si inchioda al 66% dal primo round
        # for param in efficientnet.parameters():
        #     param.requires_grad = False
        # efficientnet.classifier = torch.nn.Sequential(
        #     # torch.nn.Dropout(p=0.2, inplace=True),
        #     torch.nn.Linear(1280, efficientnet.classifier[1].in_features),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.Linear(efficientnet.classifier[1].in_features, self.classes_number))

        # print(efficientnet.classifier)

        # efficientnet.classifier[1] = torch.nn.Sequential(
        #     torch.nn.Linear(2048, 128),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.Linear(128, 2))

        # Re-init output linear layer with the right number of classes
        # efficentnet_classes_classes = efficientnet.classifier[1].in_features
        # if self.classes_number != efficentnet_classes_classes:
        #     efficientnet.classifier[1] = torch.nn.Linear(efficentnet_classes_classes, self.classes_number)
        return efficientnet

    @classmethod
    def printLog(cls, msg):
        print(msg)
        logging.log(logging.INFO, msg)
