import warnings
import requests
import torch
import torch.optim as optim
from torchvision import transforms
from collections import OrderedDict
from typing import List
import numpy as np
import flwr as fl
from src.utils.Configuration import Configuration
from src.utils.Utils import Utils
import pickle
import os
from tqdm import tqdm


from src.utils.PoisonedPartitionDataset import PoisonedPartitionDataset
from src.utils.globalVariable import blockchainApiPrefix,blockchainPrivateKeys

import time

warnings.filterwarnings('ignore')


class FlowerClient(fl.client.NumPyClient):
    BATCH_SIZE = 64

    def __init__(self, configuration: Configuration, cid: int,blockchainPrivateKey = None) -> None:
        self.cid = cid
        Utils.printLog(f'initializing client{self.cid}')
        self.configuration = configuration
        self.model = Utils.get_model(self.configuration.dataset_name,self.configuration.classes_number)
        self.epochs = 1
        self.current_round = 0
        if self.configuration.dataset_name == 'cifar10':
            self.epochs = 12
        if self.configuration.dataset_name == 'cifar100':
            self.epochs = 20

        if self.configuration.blockchain:
            self.blockchain_credential = blockchainPrivateKeys[self.cid]

    ########################################################################################
    # federated client model fit step.
    # return updated model parameters
    ########################################################################################
    def fit(self, parameters, config):
        self.current_round = config['currentRound']
        self.set_parameters(parameters)
        
        data_loader = self.get_train_data_loader()
        self.train(data_loader=data_loader)
        
        torch.cuda.empty_cache()
        torch.cuda.memory_allocated()
        
        if self.configuration.blockchain:
            self.write_parameters_on_blockchain()
            
        getted_train_parameters =  self.get_parameters(config={})
        return getted_train_parameters, len(self.get_train_data_loader()), {}

    ########################################################################################
    # Return model parameters as a list of NumPy ndarrays
    ########################################################################################
    def get_parameters(self,config) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    ########################################################################################
    # Writes into blockchian the file containing the weights obtained in the current round 
    # by the current client
    ########################################################################################
    def write_parameters_on_blockchain(self):
        path = f"./data/clientParameters/python/client{self.cid}_round{self.current_round}_parameters.pth"
        torch.save(self.model.state_dict(),path)
        with open(path, 'rb') as f:
            time.sleep(10)
            requests.post(f'{blockchainApiPrefix}/write/weights/{self.cid}/{self.current_round}',
                              data={'blockchainCredential': self.blockchain_credential},
                                files={"weights": f})


    ########################################################################################
    # Set model parameters from a list of NumPy ndarrays
    ########################################################################################
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    ########################################################################################
    # get a dataset train partition of current client from file
    ########################################################################################
    def load_train_data_from_file(self, set_transforms=False):
        train_data: torch.utils.data.dataset.Subset

        file = open(os.path.join(f"data/partitions/{self.configuration.dataset_name}",
                                 f"partition_{self.cid}.pickle"), "rb")
        try:
            train_data = pickle.load(file)
        finally:
            file.close()

        if set_transforms:
            stats = ((0.5), (0.5)) if self.configuration.dataset_name == 'mnist' else (
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(*stats),

            ])
            train_data.dataset.transform = train_transform
        return train_data

    ########################################################################################
    # obtain a dataloader of dataset train partition of current client
    ########################################################################################
    def get_train_data_loader(self):
        data = self.load_train_data_from_file(set_transforms=True)
        poisoning = self.configuration.poisoning and (self.cid in Configuration.POISONERS_CLIENTS_CID)
        if poisoning:
            data = PoisonedPartitionDataset(data, self.configuration.classes_number)
        returnValue = torch.utils.data.DataLoader(
            data,
            batch_size=FlowerClient.BATCH_SIZE, 
            shuffle=True)
        return returnValue

    ########################################################################################
    # federated client train algorithm
    ########################################################################################   
    def train(self,data_loader):
        Utils.printLog(f"client {self.cid} starting training...")

        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        self.model = self.model.to(device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=3e-3, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        train_acces = []
        train_losses = []
        self.model.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            running_corrects = 0
            total = 0
            progress_bar = tqdm(enumerate(data_loader), total=len(data_loader))

            for batch_idx, (inputs, labels) in progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                inputs = inputs.float()

                optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)
                total += labels.size(0)
                progress_bar.set_description(
                    f'Epoch [{epoch + 1}/{self.epochs}], Train Loss: {running_loss / (batch_idx + 1):.4f}, Train Acc: {100. * running_corrects / total:.2f}%')
            
            scheduler.step()

            epoch_loss = running_loss / len(data_loader)
            epoch_acc = running_corrects.double() / len(self.load_train_data_from_file())

            train_acces.append(epoch_acc * 100)
            train_losses.append(epoch_loss)

            print(f'\ntrain-loss: {np.mean(train_losses):.4f}, train-acc: {train_acces[-1]:.4f}')
       
        self.model = self.model.to('cpu')

    ########################################################################################
    # federated client model evaluate step.
    # return result of evaluate
    ########################################################################################
    def evaluate(self, parameters, config):
        # Not implemented because only centralized evaluation is used
        pass


def get_client_fn(configuration):
    def client_fn(cid: str) -> fl.client.Client:
        return FlowerClient(configuration, int(cid))

    return client_fn
