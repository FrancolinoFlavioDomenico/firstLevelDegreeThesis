from typing import Dict, List, Optional, Tuple
import flwr as fl
from FlowerClient import get_client_fn
from flwr.simulation.ray_transport.utils import enable_tf_gpu_growth
from flwr.common import Metrics
import numpy as np
from sklearn.metrics import confusion_matrix

import Plotter
import globalVariable as gv
import ModelConf as mf


class Server:

    def __init__(self, model_conf: mf.ModelConf) -> None:
        self.model_conf = model_conf
        self.strategy = fl.server.strategy.FedAvg(
            min_fit_clients=gv.CLIENTS_NUM,
            min_evaluate_clients=gv.CLIENTS_NUM,
            min_available_clients=gv.CLIENTS_NUM,
            evaluate_fn=self.get_eval_fn(),
            evaluate_metrics_aggregation_fn=Server.weighted_average,
            # initial_parameters=fl.common.ndarrays_to_parameters(self.model_conf.resnet_model.get_weights())
        )
        self.accuracy_data = []
        self.loss_data = []
        
        self.plotter = Plotter.Plotter(self.model_conf.dataset_name, model_conf.poisoning,model_conf.blockchain)

    def start_simulation(self):
        client_resources = {"num_cpus": 2, "num_gpus": 0.5}
        fl.simulation.start_simulation(
            client_fn=get_client_fn(self.model_conf),
            num_clients=gv.CLIENTS_NUM,
            config=fl.server.ServerConfig(num_rounds=gv.ROUNDS_NUM),
            strategy=self.strategy,
            client_resources=client_resources,
            actor_kwargs={
                "on_actor_init_fn": enable_tf_gpu_growth,
            }
        )

    @staticmethod
    def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        # Aggregation function for (federated) evaluation metrics, i.e. those returned by
        # the client's evaluate() method.
        # Multiply accuracy of each client by number of examples used
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]

        # Aggregate and return custom metric (weighted average)
        return {"accuracy": sum(accuracies) / sum(examples)}

    def get_eval_fn(self):
        model_conf = self.model_conf

        def evaluate(
                server_round: int,
                parameters: fl.common.NDArrays,
                config: Dict[str, fl.common.Scalar],
        ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
            print(f'round {server_round} of dataset {self.model_conf.dataset_name}')
            model = model_conf.get_model()
            model.set_weights(parameters)  # Update model with the latest parameters
            loss, accuracy = model.evaluate(model_conf.x_test, model_conf.y_test)
            self.accuracy_data.append(accuracy)
            self.loss_data.append(loss)
            if server_round == gv.ROUNDS_NUM:
                self.plotter.line_chart_plot(self.accuracy_data, self.loss_data)
                Server.set_confusion_matrix(model, model_conf, self.plotter)
               

            return loss, {"accuracy": accuracy}

        return evaluate
    
    @staticmethod
    def set_confusion_matrix(model, model_conf,plotter):
            y_predict = model.predict(model_conf.x_test)
            y_predict = np.argmax(y_predict, axis=1)
            y_test = np.argmax(model_conf.y_test,axis=1)
            result = confusion_matrix(y_test,y_predict)
            plotter.confusion_matrix_chart_plot(result)