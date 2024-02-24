import flwr as fl
from model import Model

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