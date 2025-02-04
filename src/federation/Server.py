import warnings
from typing import Dict, List, Optional, Tuple
import flwr as fl
from flwr.common import Metrics
import requests
from sklearn.metrics import confusion_matrix
from src.utils.Utils import Utils
from src.utils.Configuration import Configuration
import torch
from torch.utils.data.dataloader import DataLoader
from collections import OrderedDict
from src.federation.FlowerClient import get_client_fn
from src.utils.globalVariable import blockchainApiPrefix,blockchainPrivateKeys
from src.federation.FedAVGcustom import FedAVGcustom
from src.plotting.Plotter import Plotter as plt

warnings.filterwarnings('ignore')

class Server:

    def __init__(self, configuration: Configuration) -> None:
        self.configuration = configuration

        self.model = Utils.get_model(self.configuration.dataset_name,self.configuration.classes_number)

        self.accuracy_data = []
        self.loss_data = []

        if self.configuration.blockchain:
            self.blockchain_credential = blockchainPrivateKeys[Configuration.CLIENTS_NUM]
            requests.post(f'{blockchainApiPrefix}/deploy/contract',
                json={'blockchainCredential': self.blockchain_credential})
            
            self.strategy = FedAVGcustom(
                    min_fit_clients=Configuration.CLIENTS_NUM,
                    min_evaluate_clients=0,
                    min_available_clients=Configuration.CLIENTS_NUM,
                    evaluate_fn=self.get_evaluate_fn(),
                    fraction_evaluate=0,
                    on_fit_config_fn=self.get_fit_config,
                    dataset_name = self.configuration.dataset_name,  
                    classes_number = self.configuration.classes_number  
                )
        else:
            self.strategy = fl.server.strategy.FedAvg(
                min_fit_clients=self.configuration.CLIENTS_NUM,
                min_evaluate_clients=0,
                min_available_clients=self.configuration.CLIENTS_NUM,
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
        test_data_loader = DataLoader(Utils.get_test_data(self.configuration.dataset_name))

        def evaluate(
                server_round: int,
                parameters: fl.common.NDArrays,
                config: Dict[str, fl.common.Scalar],
        ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
            # Update model with the latest parameters
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)

            loss, accuracy = Utils.test(self.model,test_data_loader)

            # add chart data
            self.loss_data.append(loss)
            self.accuracy_data.append(accuracy)
            if server_round == Configuration.ROUNDS_NUMBER:  #last round
                plt.line_chart_plot(self.accuracy_data, self.loss_data)
                plt.line_chart_plot_grouped(self.accuracy_data, self.loss_data,server_round)
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
        plt.confusion_matrix_chart_plot(result)

    ########################################################################################
    # start federated simulation
    ########################################################################################
    def start_simulation(self):
        client_resources = {"num_cpus": 2, "num_gpus": 0.5}
        fl.simulation.start_simulation(
            client_fn=get_client_fn(self.configuration),
            num_clients=Configuration.CLIENTS_NUM,
            config=fl.server.ServerConfig(num_rounds=Configuration.ROUNDS_NUMBER),
            strategy=self.strategy,
            client_resources=client_resources,
        )
