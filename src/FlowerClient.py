import flwr as fl
import numpy as np
import gc
from secml.adv.attacks import CAttackPoisoning
from secml.array.c_array import CArray
from secml.data import CDataset
from secml.data.loader import CDataLoaderPyTorch, CDataLoader
import copy

import ModelConf
import globalVariable as gv

from src import logger


class FlowerClient(fl.client.NumPyClient):
    EPOCHS = 10
    BATCH_SIZE = 20
    STEPS_FOR_EPOCHS = 3
    VERBOSE = 0

    def __init__(self, model_conf: ModelConf.ModelConf, cid, client_partition_training_data=None) -> None:
        self.model_conf = model_conf
        (x_train, y_train) = client_partition_training_data
        self.x_train = copy.deepcopy(x_train)

        if model_conf.poisoning and (cid in gv.POISONERS_CLIENTS_CID):
            print(f'client {cid} starting label flipping poisoning')
            logger.info(f'client {cid} starting label flipping poisoning')
            self.y_train = np.random.permutation(y_train)
            self.run_secml_poisoning()
        else:
            self.y_train = copy.deepcopy(y_train)

        del x_train
        del y_train
        gc.collect()

        self.model = self.model_conf.get_model()
    
    def run_secml_poisoning(self):
        for item in self.x_train:
            item = self.add_perturbation(item)


    def add_perturbation(self,img):
        scale = 0.8
        rows, cols, channels = img.shape
        # Create noise array with the same shape as the image
        noise = np.zeros_like(img)

        #if noise_type == "random":
        noise += np.random.rand(rows, cols, channels) * scale
        #elif noise_type == "gaussian":
        #noise += np.random.normal(0, scale, size=img.shape)

        # Clip noise values to be within image value range (usually 0-255)
        perturbed_img = np.clip(img + noise, 0, 255)
        return perturbed_img
            
            
    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=FlowerClient.EPOCHS, batch_size=FlowerClient.BATCH_SIZE,
                       steps_per_epoch=FlowerClient.STEPS_FOR_EPOCHS)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.model_conf.x_test, self.model_conf.y_test,
                                             verbose=FlowerClient.VERBOSE)
        return loss, len(self.model_conf.x_test), {"accuracy": float(accuracy)}


def get_client_fn(model_conf):
    def client_fn(cid: str) -> fl.client.Client:
        return FlowerClient(model_conf, int(cid), model_conf.get_client_training_partitions_of(int(cid)))

    return client_fn
