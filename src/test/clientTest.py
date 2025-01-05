# TODO remove this class
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torchvision import transforms
import numpy as np
import flwr as fl

from src.utils import Utils
import pickle
import os
from collections import OrderedDict

from tqdm import tqdm
from src.utils.PoisonedPartitionDataset import PoisonedPartitionDataset

from src.utils.globalVariable import seed_value
torch.manual_seed(seed_value)


class FlowerClient(fl.client.NumPyClient):
    BATCH_SIZE = 128

    def __init__(self, utils: Utils, cid: int) -> None:
        self.cid = cid
        Utils.printLog(f'initializing client{self.cid}')
        self.utils = utils
        self.model = self.utils.get_model()
        self.epochs = 5 if self.utils.dataset_name != 'mnist' else 2

    def load_train_data_from_file(self, set_transforms=False):
        train_data: torch.utils.data.dataset.Subset

        file = open(os.path.join(f"data/partitions/{self.utils.dataset_name}",
                                 f"partition_{self.cid}.pickle"), "rb")
        try:
            train_data = pickle.load(file)
        finally:
            file.close()

        if set_transforms:
            stats = ((0.5), (0.5)) if self.utils.dataset_name == 'mnist' else (
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(*stats),

            ])
            train_data.dataset.transform = train_transform
        return train_data

    def get_train_data_loader(self):
        data = self.load_train_data_from_file(set_transforms=True)
        returnValue = torch.utils.data.DataLoader(
            PoisonedPartitionDataset(data, self.utils.classes_number) if self.utils.poisoning else data,
            batch_size=FlowerClient.BATCH_SIZE, shuffle=False)
        return returnValue

    def run_poisoning(self, data_loader):
        self.utils.printLog(f'client {self.cid} starting poisoning')
        loader = data_loader
        for batch in loader:
            img, labels = batch
            labels = labels[torch.randperm(labels.size(0))]  # label flipping
            img = self.add_perturbation(img)
        return loader

    def add_perturbation(self, img):
        scale = 0.8
        noise = torch.randn(img.shape) * scale
        perturbed_img = img + noise
        return perturbed_img

    def test(self):
        data_loader = self.get_train_data_loader()

        progress_bar = tqdm(enumerate(data_loader), total=len(data_loader))

        for batch_idx, (inputs, labels) in progress_bar:
            if batch_idx == 1:
                print("label lenng", labels[0])
                self.imshow(inputs[0])
                plt.title(labels[0])
                print(
                    "------------------------------------------------------------------------------------------------------------------------")


    def imshow(self, img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
