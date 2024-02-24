from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Dropout
from keras.layers import MaxPooling2D

from keras.utils import to_categorical

import flwr as fl
from model import Model
from typing import Dict, Optional, Tuple


import flwr as fl
from flwr.simulation.ray_transport.utils import enable_tf_gpu_growth

from keras.datasets import mnist
from keras.datasets import cifar10
from keras.datasets import cifar100


class Model:        
    
    CLIENTS_NUM = 2
    ROUNDS_NUM = 2
    MODEL = None
    CLASSESS_NUMBER = 10
    DATASET = cifar10
    KERNEL_SIZE = (4,4)
    INPUT_SHAPE = (32,32,3)
    
    X_TRAIN = None
    Y_TRAIN = None
    X_TEST = None
    Y_TEST = None
    
    
    @classmethod
    def setData(cls, dataSet = None, classes_number = None, kernel_size = None, input_shape = None):
        if(dataSet is not None):
            cls.DATASET = dataSet
            cls.CLASSESS_NUMBER = classes_number
            cls.KERNEL_SIZE = kernel_size
            cls.INPUT_SHAPE = input_shape
            cls.DATASET = dataSet
        #load data
        (cls.X_TRAIN, cls.Y_TRAIN), (cls.X_TEST, cls.Y_TEST) = cls.DATASET.load_data()
        #normalizing e label encodig
        cls.X_TRAIN = cls.X_TRAIN/255
        cls.X_TEST = cls.X_TEST/255
        cls.Y_TRAIN = to_categorical(cls.Y_TRAIN, classes_number)
        cls.Y_TEST = to_categorical(cls.Y_TEST, classes_number)
        
    
    @classmethod
    def setModel(cls):
        #build the model
        cls.MODEL = Sequential()
        cls.MODEL.add(Conv2D(64,cls.KERNEL_SIZE,input_shape=cls.INPUT_SHAPE,activation="relu"))
        cls.MODEL.add(MaxPooling2D(pool_size=(2,2)))
        cls.MODEL.add(Dropout(0.5))
        cls.MODEL.add(Conv2D(64,cls.KERNEL_SIZE,input_shape=cls.INPUT_SHAPE,activation="relu"))
        cls.MODEL.add(MaxPooling2D(pool_size=(2,2)))
        cls.MODEL.add(Dropout(0.25))
        cls.MODEL.add(Flatten())
        cls.MODEL.add(Dense(256,activation="relu"))
        cls.MODEL.add(Dense(cls.CLASSESS_NUMBER,activation="softmax"))
            
        cls.MODEL.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
        
       
       


class Client(fl.client.NumPyClient):    
    
    EPOCHS = 10
    BATCH_SIZE = 130
    STEPS_FOR_EPOCHS = 3
    VERBOSE = 0
    
    def __init__(self, x_train, y_train, x_test, y_test) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model =  Model.MODEL
        print("client init")
        
    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        print("fitting client")
        self.model.set_weights(parameters)
        self.model.fit(self.x_train,self.y_train, epochs=self.EPOCHS, batch_size=self.BATCH_SIZE, steps_per_epoch=self.STEPS_FOR_EPOCHS)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        print("evaluate client")   
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test,verbose=self.VERBOSE)
        return loss, len(self.x_test), {"accuracy": float(accuracy)}
    
    
    



def get_client_fn(x_train, y_train, x_test, y_test):
    
    """Return a function to construct a client.

    The VirtualClientEngine will execute this function whenever a client is sampled by
    the strategy to participate.
    """
    #print(f"client class is {Client(x_train, y_train, x_test, y_test)}")
    #print(Client(m.Model.X_TRAIN, m.Model.Y_TRAIN, m.Model.X_TEST, m.Model.Y_TEST))

    def client_fn(cid: str) -> fl.client.Client:
        """Construct a FlowerClient with its own dataset partition."""

        # Extract partition for client with id = cid
        # client_dataset = dataset.load_partition(int(cid), "train")

        # # Now let's split it into train (90%) and validation (10%)
        # client_dataset_splits = client_dataset.train_test_split(test_size=0.1)

        # trainset = client_dataset_splits["train"].to_tf_dataset(
        #     columns="image", label_cols="label", batch_size = Cifar10Client.BATCH_SIZE
        # )
        # valset = client_dataset_splits["test"].to_tf_dataset(
        #     columns="image", label_cols="label", batch_size = Cifar10Client.BATCH_SIZE
        # )
        
        #trainset, valset = ModelClass.getData()
        
        # Create and return client
        
        #print(Client(m.Model.X_TRAIN, m.Model.Y_TRAIN, m.Model.X_TEST, m.Model.Y_TEST))
        return Client(x_train, y_train, x_test, y_test)

    return client_fn







def get_eval_fn():
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
            model = Model.MODEL
            model.set_weights(parameters)  # Update model with the latest parameters
            loss, accuracy = model.evaluate(Model.X_TEST,Model.Y_TEST)
            print("After round {}, Global accuracy = {} ".format(server_round,accuracy))
            """results = {"round":server_round,"loss": loss, "accuracy": accuracy}
            results_list.append(results) """
            return loss, {"accuracy": accuracy}

        return evaluate
    
    
    
    
    
    
         


# Enable GPU growth in main process
enable_tf_gpu_growth()
client_resources = {"num_cpus": 2, "num_gpus": 0.25}

#Cifar10
Model.setData(cifar10, 10, (4,4), (32, 32, 3))
Model.setModel()
#s.start_simulation(s.get_strategy())
strategy = fl.server.strategy.FedAvg(  
                            min_fit_clients = Model.CLIENTS_NUM,
                            min_evaluate_clients = Model.CLIENTS_NUM,
                            min_available_clients = Model.CLIENTS_NUM,
                            evaluate_fn = get_eval_fn()
                            )

fl.simulation.start_simulation(
    client_fn = get_client_fn(Model.X_TRAIN, Model.Y_TRAIN, Model.X_TEST, Model.Y_TEST),
    num_clients = Model.CLIENTS_NUM,
    config=fl.server.ServerConfig(num_rounds=Model.ROUNDS_NUM),
    strategy = strategy,
    client_resources=client_resources,
    # actor_kwargs={
    #      "on_actor_init_fn": enable_tf_gpu_growth,
    # }
)