import flwr as fl

from torch.utils.data import DataLoader

# import Utils

from Utils import Utils
import pickle
import os
import torch
from torch.utils.data.dataset import Subset

from collections import OrderedDict
from tqdm import tqdm

class FlowerClient(fl.client.NumPyClient):

    def __init__(self, utils: Utils, cid: int) -> None:
        self.utils = utils
        self.cid = cid
        Utils.printLog(f'initializing client{self.cid}')

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        self.model = self.utils.get_model()
        self.epochs = 1 if self.utils.dataset_name != 'mnist' else 1
        self.batch_size = 100
        self.client_partition_data: Subset
        with open(os.path.join(f"../data/partitions/{self.utils.dataset_name}",
                               f"partition_{0}.pickle"), "rb") as f:
            self.client_partition_data = pickle.load(f)

        train_data_size = int(0.9 * len(self.client_partition_data))  # 90% for training
        test_data_size = len(self.client_partition_data) - train_data_size
        self.train_data, self.test_data = torch.utils.data.random_split(self.client_partition_data,
                                                                        [train_data_size, test_data_size])

        # self.steps_for_epoch = len(self.x_train) // self.batch_size
        # self.verbose = 0

        self.train_data_loader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=True)
        self.test_data_loader = DataLoader(self.test_data, batch_size=self.batch_size)

        if utils.poisoning and (self.cid in utils.POISONERS_CLIENTS_CID):
            self.run_poisoning()


    def run_poisoning(self):
        self.utils.printLog(f'client {self.cid} starting poisoning')
        # self.y_train = np.random.permutation(self.y_train)
        for batch in self.train_data_loader:
            img, labels = batch
            labels = labels[torch.randperm(labels.size(0))]  # label flipping
            img = self.add_perturbation(img)

    def add_perturbation(self, img):
        scale = 0.8
        noise = torch.randn(img.shape) * scale
        perturbed_img = img + noise

        return perturbed_img

    def set_parameters(self, parameters):
        """Loads model and replaces it parameters with the
        ones given."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        self.set_parameters(parameters)

        # Get hyperparameters for this round
        # batch_size: int = config["batch_size"]
        # epochs: int = config["local_epochs"]

        results = self.train()

        parameters_prime = self.get_model_params()
        num_examples_train = len(self.train_data)

        return parameters_prime, num_examples_train, results

    def train(self):
        """Train the network on the training set."""
        print("Starting training...")
        self.model.to(self.device)  # move model to GPU if available
        criterion = torch.nn.CrossEntropyLoss().to(self.device)
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4
        )

        self.model.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            progress_bar = tqdm(enumerate(self.train_data_loader), total=len(self.test_data_loader))
            for batch_idx, (images, labels) in progress_bar:
                # images, labels = batch["img"], batch["label"]
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(images), labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = self.model(images).max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                progress_bar.set_description(
                    f'Epoch [{epoch + 1}/{self.epochs}], Train Loss: {running_loss / (batch_idx + 1):.4f}, Train Acc: {100. * correct / total:.2f}%')

        self.model.to("cpu")  # move model back to CPU

        train_loss, train_acc = self.test(self.train_data_loader)
        val_loss, val_acc = self.test(self.test_data_loader)

        results = {
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        }
        return results

    def test(self,data_loader):
        """Validate the network on the entire test set."""
        print(f"Starting evalutation client{self.cid}...")
        device: torch.device = torch.device("cpu")
        self.model.to(device)  # move model to GPU if available
        criterion = torch.nn.CrossEntropyLoss()
        correct, loss = 0, 0.0
        self.model.eval()
        with torch.no_grad():
            for batch in data_loader:
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
        accuracy = correct / len(data_loader.dataset)
        # print(
        #     f'Epoch [{self.epochs + 1}/{self.epochs}], Train Loss: {loss:.4f}, Train Acc: {100. * correct / total:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {accuracy:.2f}%')
        print(f'Val Loss: {loss:.4f}, Val Acc: {accuracy:.2f}%')
        # self.model.to("cpu")  # move model back to CPU
        return loss, accuracy

    def get_model_params(self):
        """Returns a model's parameters."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def evaluate(self, parameters, config):
        Utils.printLog(f'client {self.cid} evaluating model')
        """Evaluate parameters on the locally held test set."""
        # Update local model parameters
        self.set_parameters(parameters)

        # Get config values
        # steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        # testloader = DataLoader(self.testset, batch_size=16)

        loss, accuracy = self.test(self.test_data_loader)

        # print(
        #     f'Epoch [{self.epochs + 1}/{self.epochs}], Train Loss: {loss:.4f}, Train Acc: {100. * correct / total:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {accuracy:.2f}%')
        Utils.printLog(
            f'Val Loss: {loss:.4f}, Val Acc: {accuracy:.2f}%')

        return float(loss), len(self.test_data), {"accuracy": float(accuracy)}

    def start_client(self):
        # client = CifarClient(trainset, testset, device, args.model).to_client()
        fl.client.start_client(server_address="127.0.0.1:8080", client=self.to_client())




def get_client_fn(model_conf):
    def client_fn(cid: str) -> fl.client.Client:
       return FlowerClient(model_conf, int(cid))

    return client_fn


