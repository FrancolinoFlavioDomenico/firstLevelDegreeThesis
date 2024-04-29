import flwr as fl
import numpy as np
import logging
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import random

import ModelConf
import globalVariable as gv


class FlowerClient(fl.client.NumPyClient):


    def __init__(self, model_conf: ModelConf.ModelConf, cid, client_partition_training_data=None) -> None:
        self.cid = cid
        gv.printLog(f'initializing client{self.cid}')
        self.model_conf = model_conf
        self.model = self.model_conf.get_model()
        self.x_train, self.y_train = client_partition_training_data
        
        self.epochs = 25 if self.model_conf.dataset_name != 'mnist' else 5
        self.batch_size = 250 if self.model_conf.dataset_name != 'mnist' else 50
        self.steps_for_epoch = len(self.x_train) // self.batch_size
        self.verbose = 0
        
        if model_conf.poisoning and (self.cid in gv.POISONERS_CLIENTS_CID):
            self.run_poisoning()

        self.train_set_increased = self.generate_data()

        
    def generate_data(self):
        data_generator = ImageDataGenerator(rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True
            )
        return data_generator.flow(self.x_train, self.y_train,self.batch_size)
        
        
        
    def run_poisoning(self):
        gv.printLog(f'client {self.cid} starting label flipping poisoning')
        self.y_train = np.random.permutation(self.y_train)
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
        noise += np.random.normal(0, scale, size=img.shape)
        # Clip noise values to be within image value range (usually 0-255)
        perturbed_img = np.clip(img + noise, 0, 255)
        return perturbed_img
            
            
    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        gv.printLog(f'client {self.cid} fitting model')
        self.model.set_weights(parameters)        
        
       # early_stop = EarlyStopping(monitor='val_loss', patience=2)
        self.model.fit(self.train_set_increased,
            epochs=self.epochs,
            steps_per_epoch=self.steps_for_epoch,
            validation_data=(self.model_conf.x_test,self.model_conf.y_test), 
            #callbacks=[early_stop],
            #batch_size=batch_size,
            )
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        gv.printLog(f'client {self.cid} evaluating model')
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.model_conf.x_test, self.model_conf.y_test,
                                             verbose=self.verbose)
        return loss, len(self.model_conf.x_test), {"accuracy": float(accuracy)}


def get_client_fn(model_conf):
    def client_fn(cid: str) -> fl.client.Client:
       return FlowerClient(model_conf, int(cid), model_conf.get_client_training_partitions_of(int(cid)))

    return client_fn
