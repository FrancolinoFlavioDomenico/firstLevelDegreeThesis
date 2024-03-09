import flwr as fl
import ModelConf


class FlowerClient(fl.client.NumPyClient):
    EPOCHS = 10
    BATCH_SIZE = 20
    STEPS_FOR_EPOCHS = 3
    VERBOSE = 0

    def __init__(self, client_partition_training_data, model_conf: ModelConf.ModelConf) -> None:
        self.model_conf = model_conf
        (self.x_train, self.y_train) = client_partition_training_data
        self.model = self.model_conf.get_model()


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
        return FlowerClient(model_conf.get_client_training_partitions_of(int(cid)), model_conf)

    return client_fn
