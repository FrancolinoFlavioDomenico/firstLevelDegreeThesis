from keras.datasets import mnist
from keras.datasets import cifar10
from keras.datasets import cifar100
from flwr.simulation.ray_transport.utils import enable_tf_gpu_growth
import time
import gc

import ModelConf as mf
import Server
import globalVariable as gv

from src import logger


enable_tf_gpu_growth()

#####################################
#
#      simulation start function
#
#####################################


# cifar10
def start_cifa10(poisoning = False):
    print("starting cifar10 dataset")
    logger.info("starting cifar10 dataset")
    cifar10_model_conf = mf.ModelConf('cifar10', cifar10, 10, (4, 4), (32, 32, 3), poisoning)
    server_cifar10 = Server.Server(cifar10_model_conf)
    server_cifar10.start_simulation()
    del server_cifar10
    del cifar10_model_conf
    print("finish cifar10")
    logger.info("finish cifar10")

# mnist
def start_mnist(poisoning = False):
    print("starting mnist dataset")
    logger.info("starting mnist dataset")
    mnist_model_conf = mf.ModelConf('mnist', mnist, 10, (3, 3), (28, 28, 1), poisoning)
    server_mnist = Server.Server(mnist_model_conf)
    server_mnist.start_simulation()
    del server_mnist
    del mnist_model_conf
    print("finish mnist")
    logger.info("finish mnist")

# cifar100
def start_cifar100(poisoning = False):
    print("starting cifar100 dataset")
    logger.info("starting cifar100 dataset")
    cifar100_model_conf = mf.ModelConf('cifar100', cifar100, 100, (4, 4), (32, 32, 3),  poisoning)
    server_cifar100 = Server.Server(cifar100_model_conf)
    server_cifar100.start_simulation()
    del server_cifar100
    del cifar100_model_conf
    print("finish cifar100")
    logger.info("finish cifar100")


def clear_ram():
    time.sleep(20)
    gc.collect()
    time.sleep(20)

#####################################
#
#       whitout poisoning
#
#####################################
start_cifa10(False)
clear_ram()

start_mnist(False)
clear_ram()

start_cifar100(False)
clear_ram()



#####################################
#
#       whit poisoning
#
#####################################
start_cifa10(True)
clear_ram()

start_mnist(True)
clear_ram()

start_cifar100(True)
clear_ram()


