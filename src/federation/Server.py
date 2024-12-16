import warnings
from typing import Dict, List, Optional, Tuple
import flwr as fl
from flwr.common import Metrics

import requests
from sklearn.metrics import confusion_matrix

from src.plotting import Plotter
from src.utils.Utils import Utils

from torchvision import datasets
import torch
from torch.utils.data.dataloader import DataLoader
from collections import OrderedDict
from src.federation.FlowerClient import get_client_fn
from src.utils.globalVariable import blockchainApiPrefix,blockchainPrivateKeys

warnings.filterwarnings('ignore')

class Server:
    ROUNDS_NUMBER = 5
    BATCH_SIZE = 64

    def __init__(self, utils: Utils) -> None:
        # Utils.printLog('server starting')
        self.utils = utils

        self.model = utils.get_model()
        # TODO remove
        # self.model_parameters = [val.cpu().numpy() for _, val in self.model.state_dict().items()]

        self.accuracy_data = []
        self.loss_data = []

        self.plotter = Plotter.Plotter(self.utils.dataset_name, Server.ROUNDS_NUMBER, self.utils.poisoning,
                                       self.utils.blockchain)
        if self.utils.blockchain:
            self.blockchain_credential = blockchainPrivateKeys[0]
            r = requests.post(f'{blockchainApiPrefix}/deploy/contract',
                json={'blockchainCredential': self.blockchain_credential})

        #TODO custom strategy that exted fedavg and overwrite only aggregate_fit for blockchain
        # (overwrite consists into remove poisoned result and then call overided method)
        self.strategy = fl.server.strategy.FedAvg(
            min_fit_clients=self.utils.CLIENTS_NUM,
            min_evaluate_clients=0,
            min_available_clients=self.utils.CLIENTS_NUM,
            evaluate_fn=self.get_evaluate_fn(),
            fraction_evaluate=0,
            on_fit_config_fn=self.get_fit_config  
        )
        
    def get_fit_config(self, server_round:int):
        return { "currentRound":server_round}

    ########################################################################################
    # Return an evaluation function for server-side evaluation.
    ########################################################################################
    def get_evaluate_fn(self):
        test_data_loader = DataLoader(self.utils.get_test_data())

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

            # add chart data
            self.loss_data.append(loss)
            self.accuracy_data.append(accuracy)
            if server_round == Server.ROUNDS_NUMBER:  #last round
                self.plotter.line_chart_plot(self.accuracy_data, self.loss_data)
                self.set_confusion_matrix(test_data_loader)
            return loss, {"accuracy": accuracy}

        return evaluate

    ########################################################################################
    # Aggregation function for (federated) evaluation metrics, i.e. those returned by
    # the client's evaluate() method.
    # Multiply accuracy of each client by number of examples used
    ########################################################################################
    @staticmethod
    def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]

        # Aggregate and return custom metric (weighted average)
        return {"accuracy": sum(accuracies) / sum(examples)}

    ########################################################################################
    #  build a confusion matrix chart
    ########################################################################################
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

    #TODO remove?
    ########################################################################################
    # Start Flower server for n rounds of federated learning
    ########################################################################################
    # def start_server(self):
    #     print("Starting server flower...")
    #     fl.server.start_server(
    #         server_address="0.0.0.0:8080",
    #         config=fl.server.ServerConfig(num_rounds=Server.ROUNDS_NUMBER),
    #         strategy=self.strategy,
    #     )

    ########################################################################################
    # start federated simulation
    ########################################################################################
    def start_simulation(self):
        client_resources = {"num_cpus": 2, "num_gpus": 0.5}
        fl.simulation.start_simulation(
            client_fn=get_client_fn(self.utils),
            num_clients=Utils.CLIENTS_NUM,
            config=fl.server.ServerConfig(num_rounds=Server.ROUNDS_NUMBER),
            strategy=self.strategy,
            client_resources=client_resources,
        )
