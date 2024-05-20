import pickle
import os

import torch
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.transforms import RandomHorizontalFlip, RandomCrop
from torchvision.models import efficientnet_b0, resnet50, resnet18
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

        self.train_data = self.download_data(True)
        self.test_data = self.download_data(False)

        # print("------------------------------------------------------------------------------")
        # self.get_model()
        # print("------------------------------------------------------------------------------")

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

    def download_data(self, train):
        stats = ((0.5), (0.5)) if self.dataset_name == 'mnist' else ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # train_transform = Compose([
        #     # RandomHorizontalFlip(),
        #     # RandomCrop(28 if self.dataset_name == 'mnist' else 32, padding=4, padding_mode="reflect"),
        #     ToTensor(),
        #     Normalize(*stats),
        #
        # ])

        test_transform = Compose([
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

        for param in efficientnet.parameters():
            param.requires_grad = False

        # for layer in efficientnet.children():
        #     print(layer)
        layer_to_unfreeze = [param  for name, param  in efficientnet.named_parameters() if "Conv2d" in name]
        layer_to_unfreeze = layer_to_unfreeze[-4:]
        print(f"Number of layers: {len(layer_to_unfreeze)}")
        print(layer_to_unfreeze)
        for layer in layer_to_unfreeze:
            for param in layer.parameters():
                param.requires_grad = True

        # Re-init output linear layer with the right number of classes
        efficentnet_classes_classes = efficientnet.classifier[1].in_features
        if self.classes_number != efficentnet_classes_classes:
            efficientnet.classifier[1] = torch.nn.Linear(efficentnet_classes_classes, self.classes_number)
        return efficientnet

    @classmethod
    def printLog(cls, msg):
        print(msg)
        logging.log(logging.INFO, msg)
