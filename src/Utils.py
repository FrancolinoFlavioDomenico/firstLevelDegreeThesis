import pickle
import os

import torch
from torchvision.transforms import  ToTensor, Normalize, Compose
from torchvision.transforms import RandomHorizontalFlip, RandomCrop
from torchvision.models import efficientnet_b0
import warnings


import numpy as np
import logging

from torchvision import datasets

import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid



class Utils:
    CLIENTS_NUM = 10
    POISONERS_CLIENTS_CID = np.random.randint(0, CLIENTS_NUM, round((CLIENTS_NUM * 30) / 100))
    DATASET_PATH = '../data/torchDownload'

    def __init__(self, dataset_name, classes_number, kernel_size, input_shape, poisoning=False, blockchain=False):
        self.dataset_name = dataset_name
        self.classes_number = classes_number
        self.kernel_size = kernel_size
        self.input_shape = input_shape
        self.poisoning = poisoning
        self.blockchain = blockchain

        stats = ((0.5), (0.5)) if self.dataset_name == 'mnist' else ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        train_transform = Compose([
            RandomHorizontalFlip(),
            RandomCrop(32, padding=4, padding_mode="reflect"),
            ToTensor(),
            Normalize(*stats),

        ])

        if self.dataset_name == 'cifar100':
            self.train_data = datasets.CIFAR100(
                root=Utils.DATASET_PATH,
                train=True,
                download=True,
                transform=train_transform
            )
        elif self.dataset_name == 'cifar10':
            self.train_data = datasets.CIFAR10(
                root=Utils.DATASET_PATH,
                train=True,
                download=True,
                transform=train_transform
            )
        else:
            self.train_data = datasets.MNIST(
                root=Utils.DATASET_PATH,
                train=True,
                download=True,
                transform=train_transform
            )
            # serve?
            # self.train_data.data = self.train_data.data.reshape(self.train_data.data.shape[0],self.train_data.data.shape[1], self.train_data.data.shape[2],1)

        # test code.....print image with relative label
        # figure = plt.figure(figsize=(10, 8))
        # cols, rows = 5, 5
        # for i in range(1, cols * rows + 1):
        #     sample_idx = torch.randint(len(self.train_data), size=(1,)).item()
        #     img, label = self.train_data[sample_idx]
        #     figure.add_subplot(rows, cols, i)
        #     plt.title(label)
        #     plt.axis("off")
        #     plt.imshow(img, cmap="gray")
        # plt.show()

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

        # # test code.....print image of getted partition subset with relative label
        # tmp: torch.utils.data.dataset.Subset
        # with open(os.path.join(f"../data/partitions/{self.dataset_name}",
        #                        f"partition_{0}.pickle"), "rb") as f:
        #     tmp = pickle.load(f)
        # print('----------------------------------------------------------------', tmp.dataset)
        # batch_size = 128
        # train_dl = DataLoader(tmp, batch_size, num_workers=0, pin_memory=True, shuffle=True)
        # for batch in train_dl:
        #     images, labels = batch
        #     fig, ax = plt.subplots(figsize=(7.5, 7.5))
        #     ax.set_yticks([])
        #     ax.set_xticks([])
        #     ax.set_title('subset')
        #     ax.imshow(make_grid(images[:20], nrow=5).permute(1, 2, 0))
        #     break
        # plt.show()

    def generate_dataset_client_partition(self):
        partition_lenght = np.full(Utils.CLIENTS_NUM, len(self.train_data.data) / Utils.CLIENTS_NUM).astype(
            int).tolist()
        partitions = torch.utils.data.random_split(self.train_data, partition_lenght)
        # print(partitions[0].dataset.data.shape)
        # print(len(partitions))
        # print(partitions[0])

        dataset_partition_dir = f"../data/partitions/{self.dataset_name}"
        if not os.path.exists(dataset_partition_dir):
            os.makedirs(dataset_partition_dir)

        for i, partition in enumerate(partitions):
            with open(os.path.join(dataset_partition_dir, f"partition_{i}.pickle"), "wb") as f:
                pickle.dump(partition, f)

    def get_model(self) -> torch.nn.Module:
        """Loads EfficienNetB0 from TorchVision."""
        efficientnet = efficientnet_b0(pretrained=True)
        # Re-init output linear layer with the right number of classes
        efficentnet_classes_classes = efficientnet.classifier[1].in_features
        if self.classes_number != efficentnet_classes_classes:
            efficientnet.classifier[1] = torch.nn.Linear(efficentnet_classes_classes, self.classes_number)
        return efficientnet

    @classmethod
    def printLog(cls, msg):
        print(msg)
        logging.log(logging.INFO, msg)
