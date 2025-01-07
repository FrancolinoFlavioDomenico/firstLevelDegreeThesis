from federation import Server
import flwr as fl
import logging
import time
from utils import Configuration


DEFAULT_FORMATTER = logging.Formatter(
    "%(levelname)s %(name)s %(asctime)s | %(filename)s:%(lineno)d | %(message)s"
)
fl.common.logger.configure(identifier="executionLog", filename="log/federation_log.txt")

def start_server(configuration):
    server = Server.Server(configuration)
    server.start_simulation()


def start(dataset_name,classes_number,poisoning,blockchian):
    configuration = Configuration.Configuration(dataset_name, classes_number, poisoning, blockchian)
    time.sleep(5) # waiting for partition writing
    start_server(configuration)
    

if __name__ == "__main__":
    # start('mnist', 10, False, False)
    # start('mnist', 10, True, False)
    start('mnist', 10, True, True)
    
    # start('cifar10', 10, False, False)
    # start('cifar10', 10, False, False)
    
    # start('cifar100', 100, False, False)
    # start('cifar100', 100, False, False)