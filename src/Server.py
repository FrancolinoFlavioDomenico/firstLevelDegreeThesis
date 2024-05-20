from typing import Dict, List, Optional, Tuple
import flwr as fl
from flwr.common import Metrics

from sklearn.metrics import confusion_matrix

import Plotter
from Utils import Utils

from torchvision import datasets
from torchvision.transforms import ToTensor, Normalize, Compose
import torch
from torch.utils.data.dataloader import DataLoader
from collections import OrderedDict

from src.FlowerClient import get_client_fn

class Server:
    ROUNDS_NUMBER = 5
    BATCH_SIZE = 86

    def __init__(self, utils: Utils) -> None:
        self.utils = utils
        Utils.printLog('server starting')

        self.model = utils.get_model()
        self.model_parameters = [val.cpu().numpy() for _, val in self.model.state_dict().items()]

        self.accuracy_data = []
        self.loss_data = []

        self.plotter = Plotter.Plotter(self.utils.dataset_name, Server.ROUNDS_NUMBER, self.utils.poisoning,
                                       self.utils.blockchain)


        self.strategy = fl.server.strategy.FedAvg(
            # fraction_fit=1.0,
            # fraction_evaluate=1.0,
            min_fit_clients=self.utils.CLIENTS_NUM,
            min_evaluate_clients=Utils.CLIENTS_NUM,
            min_available_clients=self.utils.CLIENTS_NUM,
            evaluate_fn=self.get_evaluate_fn(),
            # on_fit_config_fn=self.fit_config(),
            # on_evaluate_config_fn=self.evaluate_config(),
            initial_parameters=fl.common.ndarrays_to_parameters(self.model_parameters),
        )



    def fit_config(self, server_round: int):
        """Return training configuration dict for each round.

        Keep batch size fixed at 32, perform two rounds of training with one local epoch,
        increase to two local epochs afterwards.
        """
        config = {
            "batch_size": Server.BATCH_SIZE,
            "local_epochs": 1 if server_round < 2 else 2,
        }
        return config

    def evaluate_config(self, server_round: int):
        """Return evaluation configuration dict for each round.

        Perform five local evaluation steps on each client (i.e., use five batches) during
        rounds one to three, then increase to ten local evaluation steps.
        """
        val_steps = 5 if server_round < 4 else 10
        return {"val_steps": val_steps}

    def get_evaluate_fn(self):
        """Return an evaluation function for server-side evaluation."""

        test_data_loader = DataLoader(self.utils.test_data, batch_size=Server.BATCH_SIZE)

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

            loss, accuracy = self.test(test_data_loader)

            self.loss_data.append(loss)
            self.accuracy_data.append(accuracy)
            if server_round == Server.ROUNDS_NUMBER:
                self.plotter.line_chart_plot(self.accuracy_data, self.loss_data)
                self.set_confusion_matrix(test_data_loader)
            return loss, {"accuracy": accuracy}
        return evaluate

    def test(self, test_data_loader):
        """Validate the network on the entire test set."""
        print("Starting evalutation...")
        device: torch.device = torch.device("cpu")
        self.model.to(device)  # move model to GPU if available
        criterion = torch.nn.CrossEntropyLoss()
        correct, loss = 0, 0.0
        self.model.eval()
        with torch.no_grad():
            for batch in test_data_loader:
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
        accuracy = correct / len(test_data_loader.dataset)
        return loss, accuracy

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
        # Start Flower server for four rounds of federated learning
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
            num_clients=self.utils.CLIENTS_NUM,
            config=fl.server.ServerConfig(num_rounds=Server.ROUNDS_NUMBER),
            strategy=self.strategy,
            client_resources=client_resources,
        )
