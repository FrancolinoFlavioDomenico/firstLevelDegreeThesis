from typing import Dict, Optional, Tuple

import flwr as fl
import Model
from FlowerClient import get_client_fn

from flwr.simulation.ray_transport.utils import enable_tf_gpu_growth


class Server:

    def __init__(self) -> None:
        self.strategy = fl.server.strategy.FedAvg(
            min_fit_clients=Model.Model.CLIENTS_NUM,
            min_evaluate_clients=Model.Model.CLIENTS_NUM,
            min_available_clients=Model.Model.CLIENTS_NUM,
            evaluate_fn=self.get_eval_fn()
        )
        #self.start_simulation()

    def start_simulation(self):
        client_resources = {"num_cpus": 2, "num_gpus": 0.25}
        # print(Client.get_client_fn(Model.Model.X_TRAIN, Model.Model.Y_TRAIN, Model.Model.X_TEST, Model.Model.Y_TEST))
        fl.simulation.start_simulation(
            client_fn=get_client_fn(Model.Model.X_TRAIN, Model.Model.Y_TRAIN, Model.Model.X_TEST, Model.Model.Y_TEST),
            num_clients=Model.Model.CLIENTS_NUM,
            config=fl.server.ServerConfig(num_rounds=Model.Model.ROUNDS_NUM),
            strategy=self.strategy,
            client_resources=client_resources,
            actor_kwargs={
                 "on_actor_init_fn": enable_tf_gpu_growth,
            }
        )

    def get_eval_fn(self):
        """Return an evaluation function for server-side evaluation."""
        # x_train, y_train, x_test, y_test = self.dataset

        """ x_train = self.xTrain
        y_train = self.yTrain """
        """ x_test = self.x_test
        y_test = self.y_test """

        # (x_train, y_train), (x_test, y_test) = ModelClass.getData()

        # The `evaluate` function will be called after every round
        def evaluate(
                server_round: int,
                parameters: fl.common.NDArrays,
                config: Dict[str, fl.common.Scalar],
        ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
            model = Model.Model.get_model()
            model.set_weights(parameters)  # Update model with the latest parameters
            loss, accuracy = model.evaluate(Model.Model.X_TEST, Model.Model.Y_TEST)
            print(f"After round {server_round}, Global accuracy = {accuracy}")
            """results = {"round":server_round,"loss": loss, "accuracy": accuracy}
            results_list.append(results) """
            return loss, {"accuracy": accuracy}

        return evaluate
