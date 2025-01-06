from utils import Utils
from federation import Server
import flwr as fl
import logging
import argparse


DEFAULT_FORMATTER = logging.Formatter(
    "%(levelname)s %(name)s %(asctime)s | %(filename)s:%(lineno)d | %(message)s"
)
fl.common.logger.configure(identifier="executionLog", filename="log/federation_log.txt")

def start_server():
    server = Server.Server(utils)
    server.start_simulation()

if __name__ == "__main__":
    utils = Utils.Utils('mnist', 10, False, False)
    start_server()

    utils = Utils.Utils('mnist', 10, True, False)
    start_server()
    
    # utils = Utils.Utils('mnist', 10, True, True)
    # start_server()

    # utils = Utils.Utils('cifar10', 10, False, False)
    # start_server()

    # utils = Utils.Utils('cifar10', 10, True, False)
    # start_server()
    
    # utils = Utils.Utils('cifar10', 10, True, True)
    # start_server()

    # utils = Utils.Utils('cifar100', 100, False, False)
    # start_server()

    # utils = Utils.Utils('cifar100', 100, True, False)
    # start_server()
    
    # utils = Utils.Utils('cifar100', 100, True, True)
    # start_server()