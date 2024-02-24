import flwr as fl
from typing import Dict, Optional, Tuple
import client as c
import model as m


class Server:
        
    def __init__(self) -> None:
        self.strategy =  fl.server.strategy.FedAvg(  
                        min_fit_clients = m.Model.CLIENTS_NUM,
                        min_evaluate_clients = m.Model.CLIENTS_NUM,
                        min_available_clients = m.Model.CLIENTS_NUM,
                        evaluate_fn = self.get_eval_fn()
                        )
        print("init server e starting simulation")
        self.start_simulation()
        
        
    def get_eval_fn(self):
        """Return an evaluation function for server-side evaluation."""
        #x_train, y_train, x_test, y_test = self.dataset
        
        """ x_train = self.xTrain
        y_train = self.yTrain """
        """ x_test = self.x_test
        y_test = self.y_test """
        
        #(x_train, y_train), (x_test, y_test) = ModelClass.getData()

        # The `evaluate` function will be called after every round
        def evaluate(
            server_round: int,
            parameters: fl.common.NDArrays,
            config: Dict[str, fl.common.Scalar],
        ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
            model = m.Model.MODEL
            model.set_weights(parameters)  # Update model with the latest parameters
            loss, accuracy = model.evaluate(m.Model.X_TEST,m.Model.Y_TEST)
            print("After round {}, Global accuracy = {} ".format(server_round,accuracy))
            """results = {"round":server_round,"loss": loss, "accuracy": accuracy}
            results_list.append(results) """
            return loss, {"accuracy": accuracy}

        return evaluate
    
    def start_simulation(self):
        print("simulation started")
        client_resources = {"num_cpus": 2, "num_gpus": 0.25}
        fl.simulation.start_simulation(
            client_fn = c.get_client_fn(m.Model.X_TRAIN, m.Model.Y_TRAIN, m.Model.X_TEST, m.Model.Y_TEST),
            num_clients = m.Model.CLIENTS_NUM,
            config=fl.server.ServerConfig(num_rounds=m.Model.ROUNDS_NUM),
            strategy = self.strategy,
            client_resources=client_resources,
            # actor_kwargs={
            #      "on_actor_init_fn": enable_tf_gpu_growth,
            # }
        )
        

    