import flwr as fl
import numpy as np
import gc
from secml.adv.attacks import CAttackPoisoning


import ModelConf
import globalVariable as gv

from src import logger

class FlowerClient(fl.client.NumPyClient):
    EPOCHS = 10
    BATCH_SIZE = 20
    STEPS_FOR_EPOCHS = 3
    VERBOSE = 0

    def __init__(self, client_partition_training_data, model_conf: ModelConf.ModelConf, cid) -> None:
        self.model_conf = model_conf
        
        (x_train, y_train) = client_partition_training_data
        if model_conf.poisoning and (cid in gv.POISONERS_CLIENTS_CID):
            print(f'client {cid} starting label flipping poisoning')
            logger.info(f'client {cid} starting label flipping poisoning')
            self.y_train = np.random.permutation(y_train)
            self.run_secml_poisoning()
        else:
            self.x_train = x_train
            self.y_train = y_train
            
        del x_train
        del y_train
        gc.collect()
        
        self.model = self.model_conf.get_model()


    def run_secml_poisoning(self):
        solver_params = {
            'eta': 0.05,
            'eta_min': 0.05,
            'eta_max': None,
            'max_iter': 100,
            'eps': 1e-6
        }
        
        seed_value = 5
        
        pois_attack = CAttackPoisoning(
            classifier=self.model,
            training_data=(self.x_train,self.y_train),
            val=(self.model_conf.x_test, self.model_conf.y_test),
            solver_params=solver_params,
            random_seed=seed_value
            )
        
        xc = self.x_train
        yc = self.y_train
        pois_attack.x0 = xc
        pois_attack.xc = xc
        pois_attack.yc = yc
        
        pois_attack.n_points = len(self.x_train) * 0.2
        pois_attack.run(self.model_conf.x_test, self.model_conf.y_test)

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=FlowerClient.EPOCHS, batch_size=FlowerClient.BATCH_SIZE,
                       steps_per_epoch=FlowerClient.STEPS_FOR_EPOCHS)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.model_conf.x_test, self.model_conf.y_test, verbose=FlowerClient.VERBOSE)
        return loss, len(self.model_conf.x_test), {"accuracy": float(accuracy)}


def get_client_fn(model_conf):

    def client_fn(cid: str) -> fl.client.Client:
        return FlowerClient(model_conf.get_client_training_partitions_of(int(cid)), model_conf, int(cid))

    return client_fn
