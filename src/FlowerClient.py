import torch
import torch.optim as optim
# from torch.utils.data.dataset import Subset
# from torch.utils.data import DataLoader
from torchvision import transforms
from collections import OrderedDict
from typing import List
import numpy as np
import flwr as fl

from Utils import Utils
import pickle
import os
from collections import OrderedDict

from tqdm import tqdm


class FlowerClient(fl.client.NumPyClient):
    
    BATCH_SIZE = 32

    def __init__(self, utils: Utils, cid: int) -> None:
        self.cid = cid
        Utils.printLog(f'initializing client{self.cid}')
        self.utils = utils
        self.model = self.utils.get_model()
        self.epochs = 5 if self.utils.dataset_name != 'mnist' else 2

        
    def fit(self, parameters, config):
        # Set model parameters, train model, return updated model parameters
        self.set_parameters(parameters)
        self.train()
        torch.cuda.empty_cache()
        torch.cuda.memory_allocated()
        return self.get_parameters(config={}), len(self.load_train_data_from_file()), {}
    
    def get_parameters(self, config) -> List[np.ndarray]:
        # Return model parameters as a list of NumPy ndarrays
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def load_train_data_from_file(self,set_transforms = False):
        train_data: torch.utils.data.dataset.Subset        
            
        file = open(os.path.join(f"data/partitions/{self.utils.dataset_name}",
                               f"partition_{self.cid}.pickle"), "rb")
        try:
            train_data = pickle.load(file)
        finally:
            file.close()

        if set_transforms:
            stats = ((0.5), (0.5)) if self.utils.dataset_name == 'mnist' else ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(*stats),

            ])
            train_data.dataset.transform = train_transform
        return train_data
    
    def get_train_data_loader(self):
        data = self.load_train_data_from_file(set_transforms=True)
        returnValue = torch.utils.data.DataLoader(data, batch_size=FlowerClient.BATCH_SIZE, shuffle=True)
        if self.utils.poisoning and (self.cid in Utils.POISONERS_CLIENTS_CID):
            returnValue = self.run_poisoning(returnValue)
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

    # def train(self):
    #     """Train the network on the training set."""
    #     print("Starting training...")
    #     device = torch.device(
    #         "cuda:0" if torch.cuda.is_available() else "cpu"
    #     )
    #     data_loader = self.get_train_data_loader()
        
    #     self.model.fc = self.model.fc.to(device)

    #     criterion = torch.nn.CrossEntropyLoss()
    #     optimizer = optim.SGD(self.model.parameters(), lr=3e-3, momentum=0.9, weight_decay=5e-4)
    #     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    #     train_acces, test_acces = [], []
    #     train_losses, test_losses = [], []
    #     total_step = len(data_loader)
    #     for epoch in range(self.epochs):
    #         print(f'Epoch {epoch}\n')

    #         running_loss = 0.0
    #         running_corrects = 0

    #         self.model.train()

    #         for batch_idx, (inputs, labels) in enumerate(data_loader):
    #             inputs = inputs.to(device)
    #             labels = labels.to(device)
    #             inputs = inputs.float()
                
    #             optimizer.zero_grad()
                
    #             outputs = self.model(inputs)
    #             loss = criterion(outputs, labels)
    #             loss.backward()
    #             optimizer.step()

    #             _, preds = torch.max(outputs, 1)
    #             running_loss += loss.item()
    #             running_corrects += torch.sum(preds == labels.data)
    #             print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(self.epoch, self.epochs-1, batch_idx, total_step, loss.item()))
    #             # if (batch_idx) % 20 == 0:
    #             #     print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch, self.epochs-1, batch_idx, total_step, loss.item()))
    #         scheduler.step()


    #         epoch_loss = running_loss / len(self.load_train_data_from_file())
    #         epoch_acc = running_corrects.double() / len(self.load_train_data_from_file())
            
    #         train_acces.append(epoch_acc * 100)
    #         train_losses.append(epoch_loss)

    #         print(f'\ntrain-loss: {np.mean(train_losses):.4f}, train-acc: {train_acces[-1]:.4f}')
    
    def train(self):
        """Train the network on the training set."""
        print("Starting training...")
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        data_loader = self.get_train_data_loader()
        
        self.model = self.model.to(device)
        # self.model.fc = self.model.fc.to(device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=3e-3, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        train_acces, test_acces = [], []
        train_losses, test_losses = [], []
        for epoch in range(self.epochs):
            print(f'Epoch {epoch}\n')

            running_loss = 0.0
            running_corrects = 0
            total = 0

            self.model.train()

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
                # print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(self.epoch, self.epochs-1, batch_idx, total_step, loss.item()))
                progress_bar.set_description(
                    f'Epoch [{epoch + 1}/{self.epochs}], Train Loss: {running_loss / (batch_idx + 1):.4f}, Train Acc: {100. * running_corrects / total:.2f}%')
                # if (batch_idx) % 20 == 0:
                #     print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch, self.epochs-1, batch_idx, total_step, loss.item()))
            scheduler.step()


            epoch_loss = running_loss / len(self.load_train_data_from_file())
            epoch_acc = running_corrects.double() / len(self.load_train_data_from_file())
            
            train_acces.append(epoch_acc * 100)
            train_losses.append(epoch_loss)

            print(f'\ntrain-loss: {np.mean(train_losses):.4f}, train-acc: {train_acces[-1]:.4f}')
        self.model = self.model.to('cpu')
        

    def evaluate(self, parameters, config):
        # Set model parameters, evaluate model on local test dataset, return result
        Utils.printLog(f'client {self.cid} evaluating model')
        self.set_parameters(parameters)
        loss, accuracy = self.utils.test(self.model)
        return float(loss), len(self.utils.get_test_data()), {"accuracy": float(accuracy)}

    def start_client(self):
        # client = CifarClient(trainset, testset, device, args.model).to_client()
        fl.client.start_client(server_address="127.0.0.1:8080", client=self.to_client())


def get_client_fn(model_conf):
    def client_fn(cid: str) -> fl.client.Client:
        return FlowerClient(model_conf, int(cid))

    return client_fn