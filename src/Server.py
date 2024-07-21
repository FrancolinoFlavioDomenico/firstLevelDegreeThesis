from typing import Dict, List, Optional, Tuple
import flwr as fl
from flwr.common import Metrics

import requests
from sklearn.metrics import confusion_matrix

import Plotter
from Utils import Utils

from torchvision import datasets
import torch
from torch.utils.data.dataloader import DataLoader
from collections import OrderedDict
from FlowerClient import get_client_fn

class Server:
    ROUNDS_NUMBER = 5
    BATCH_SIZE = 64

    def __init__(self, utils: Utils) -> None:
        Utils.printLog('server starting')
        self.utils = utils

        self.model = utils.get_model()
        self.model_parameters = [val.cpu().numpy() for _, val in self.model.state_dict().items()]

        self.accuracy_data = []
        self.loss_data = []

        self.plotter = Plotter.Plotter(self.utils.dataset_name, Server.ROUNDS_NUMBER, self.utils.poisoning,
                                       self.utils.blockchain)
        
        if self.utils.blockchain:
            response = requests.get('http://localhost:3000/getBlockchainAddress/0')
            self.blockchain_adress = response.text
            print(f"server FUNZIONAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA {self.blockchain_adress}")
            


        self.strategy = fl.server.strategy.FedAvg(
            min_fit_clients=self.utils.CLIENTS_NUM,
            min_evaluate_clients=Utils.CLIENTS_NUM,
            min_available_clients=self.utils.CLIENTS_NUM,
            evaluate_fn=self.get_evaluate_fn(),
        )


    def get_evaluate_fn(self):
        #Return an evaluation function for server-side evaluation.

        test_data_loader =  DataLoader(self.utils.get_test_data())

        # The `evaluate` function will be called after every round
        def evaluate(
                server_round: int,
                parameters: fl.common.NDArrays,
                config: Dict[str, fl.common.Scalar],
        ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
            # Update model with the latest parameters
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)

            loss, accuracy = self.utils.test(self.model)

            self.loss_data.append(loss)
            self.accuracy_data.append(accuracy)
            if server_round == Server.ROUNDS_NUMBER:
                self.plotter.line_chart_plot(self.accuracy_data, self.loss_data)
                self.set_confusion_matrix(test_data_loader)
            return loss, {"accuracy": accuracy}
        return evaluate

    @staticmethod
    def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        # Aggregation function for (federated) evaluation metrics, i.e. those returned by
        # the client's evaluate() method.
        # Multiply accuracy of each client by number of examples used
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]

        # Aggregate and return custom metric (weighted average)
        return {"accuracy": sum(accuracies) / sum(examples)}

    def set_confusion_matrix(self, test_data_loader: torch.utils.data.DataLoader):
        y_true = []
        y_pred = []
        for inputs, labels in test_data_loader:
            output = self.model(inputs)  # Feed Network

            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred.extend(output)  # Save Prediction

            labels = labels.data.cpu().numpy()
            y_true.extend(labels)  # Save Truth

        result = confusion_matrix(y_true, y_pred)
        self.plotter.confusion_matrix_chart_plot(result)

    def start_server(self):
        # Start Flower server for n rounds of federated learning
        print("Starting server flower...")
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=Server.ROUNDS_NUMBER),
            strategy=self.strategy,
        )
        
        
    def start_simulation(self):
        client_resources = {"num_cpus": 2, "num_gpus": 0.5}
        fl.simulation.start_simulation(
            client_fn=get_client_fn(self.utils),
            num_clients=Utils.CLIENTS_NUM,
            config=fl.server.ServerConfig(num_rounds=Server.ROUNDS_NUMBER),
            strategy=self.strategy,
            client_resources=client_resources,
        )
