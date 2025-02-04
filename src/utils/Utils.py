import pickle
import os
from torchvision.models import resnet50, resnet18
import torch
from typing import Tuple
import logging
from flwr.common.logger import log

logging.basicConfig(
    filename="log/simple_python_log.txt",
    encoding="utf-8",
    filemode="a",
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
)

class Utils:
    
    ########################################################################################
    # function used Validate the model on the entire test set.
    ########################################################################################
    @classmethod
    def test(
            cls,
            model,
            test_loader
    ) -> Tuple[float, float]:
        
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        criterion = torch.nn.CrossEntropyLoss()
        correct = 0
        loss = 0.0
        model.to(device)
        with torch.no_grad():
            model.eval()
            for data in test_loader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                loss += criterion(outputs, labels).item()
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        accuracy = correct / len(test_loader.dataset)
        loss = loss / len(test_loader)
        model.to('cpu')
        return loss, accuracy

    ########################################################################################
    # function for obtain a dataset test paritition used by server o client into validate step
    ########################################################################################
    @classmethod
    def get_test_data(cls,dataset_name):

        file = open(os.path.join(f"data/partitions/{dataset_name}",
                                 f"test_data.pickle"), "rb")
        try:
            test_data = pickle.load(file)
        finally:
            file.close()

        return test_data

    ########################################################################################
    # get a model arch
    ########################################################################################
    @classmethod
    def get_model(cls,dataset_name,classes_number) -> torch.nn.Module:
        if dataset_name == 'mnist':
            model = resnet18(pretrained=False)
            model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            num_ftrs = model.fc.in_features
            model.fc = torch.nn.Linear(num_ftrs, classes_number)
        else:
            model = resnet50(weights='IMAGENET1K_V1')
            num_ftrs = model.fc.in_features
            model.fc = torch.nn.Linear(num_ftrs, 256)
            model.fc = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Linear(num_ftrs, classes_number)
            )
        return model

    @classmethod
    def printLog(cls, msg, level=logging.INFO):
        logging.log(level, msg)
        log(level, msg)
