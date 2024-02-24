import flwr as fl
from client import Client

def get_client_fn(x_train, y_train, x_test, y_test):
    """Return a function to construct a client.

    The VirtualClientEngine will execute this function whenever a client is sampled by
    the strategy to participate.
    """

    def client_fn(cid: str) -> fl.client.Client:
        
        print("dioporco")
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
        print("porcosio")
        return Client(x_train, y_train, x_test, y_test)

    return client_fn