from typing import Dict, Optional, Tuple
import flwr as fl
from FlowerClient import get_client_fn
from flwr.simulation.ray_transport.utils import enable_tf_gpu_growth

import Plotter
import globalVariable as gv
import ModelConf as mf

class Server:

    def __init__(self, model_conf: mf.ModelConf) -> None:
        self.model_conf = model_conf
        self.strategy = fl.server.strategy.FedAvg(
            min_fit_clients= gv.CLIENTS_NUM,
            min_evaluate_clients=gv.CLIENTS_NUM,
            min_available_clients=((gv.CLIENTS_NUM * 50)/100),
            evaluate_fn=self.get_eval_fn()
        )
        self.plotter = Plotter.Plotter(self.model_conf.dataset_name, model_conf.poisoning)
        

    def start_simulation(self):
        client_resources = {"num_cpus": 4, "num_gpus": 0.5}
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

    def get_eval_fn(self):
        model_conf = self.model_conf
        
        def evaluate(
                server_round: int,
                parameters: fl.common.NDArrays,
                config: Dict[str, fl.common.Scalar],
        ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
            model = model_conf.get_model()
            model.set_weights(parameters)  # Update model with the latest parameters
            loss, accuracy = model.evaluate(model_conf.x_test, model_conf.y_test)
            self.plotter.accuracy_data.append(accuracy)
            self.plotter.loss_data.append(loss)
            if server_round == gv.ROUNDS_NUM:
                self.plotter.plot()
            # print(f"After round {server_round}, Global accuracy = {accuracy}")
            # results = {"round":server_round,"loss": loss, "accuracy": accuracy}
            # results_list.append(results)
            return loss, {"accuracy": accuracy}

        return evaluate
