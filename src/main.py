from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import cifar100
from flwr.simulation.ray_transport.utils import enable_tf_gpu_growth
import flwr as fl
import logging
import time
import gc

import ModelConf as mf
import Server
import globalVariable as gv

DEFAULT_FORMATTER = logging.Formatter(
"%(levelname)s %(name)s %(asctime)s | %(filename)s:%(lineno)d | %(message)s"
)
fl.common.logger.configure(identifier="executionLog", filename="log.txt")
enable_tf_gpu_growth()


#####################################
#
#      simulation start function
#
#####################################

# mnist
def start_mnist(poisoning=False,blockchain=False):
    gv.printLog(f"starting mnist dataset {'poisoned' if poisoning else ''}")
    mnist_model_conf = mf.ModelConf('mnist', mnist, 10, (3, 3), (28, 28, 1), poisoning,blockchain)
    server_mnist = Server.Server(mnist_model_conf)
    server_mnist.start_simulation()
    del server_mnist
    del mnist_model_conf
    gv.printLog(f"finish mnist {'poisoned' if poisoning else ''}")
    
    
# cifar10
def start_cifa10(poisoning=False,blockchain=False):
    gv.printLog(f"starting cifar10 dataset {'poisoned' if poisoning else ''}")
    cifar10_model_conf = mf.ModelConf('cifar10', cifar10, 10, (5, 5), (32, 32, 3), poisoning,blockchain)
    server_cifar10 = Server.Server(cifar10_model_conf)
    server_cifar10.start_simulation()
    del server_cifar10
    del cifar10_model_conf
    gv.printLog(f"finish cifar10 {'poisoned' if poisoning else ''}")


# cifar100
def start_cifar100(poisoning=False,blockchain=False):
    gv.printLog(f"starting cifar100 dataset {'poisoned' if poisoning else ''}")
    cifar100_model_conf = mf.ModelConf('cifar100', cifar100, 100, (5, 5), (32, 32, 3), poisoning,blockchain)
    server_cifar100 = Server.Server(cifar100_model_conf)
    server_cifar100.start_simulation()
    del server_cifar100
    del cifar100_model_conf
    gv.printLog(f"finish cifar100 {'poisoned' if poisoning else ''}")     


def clear_ram():
    time.sleep(1)
    gc.collect()
    time.sleep(1)


#####################################
#
#       no poisoning
#
#####################################
# start_mnist(False)
# clear_ram()

# start_cifa10(False)
# clear_ram()

start_cifar100(False)
clear_ram()


#####################################
#
#        poisoning
#
#####################################
# start_mnist(True)
# clear_ram()

# start_cifa10(True)
# clear_ram()

# start_cifar100(True)
# clear_ram()


#####################################
#
#       no poisoning blockchian
#
#####################################
# start_mnist(False,True)
# clear_ram()

# start_cifa10(False, True)
# clear_ram()

# start_cifar100(False, True)
# clear_ram()


#####################################
#
#       poisoning blockchian
#
#####################################
# start_mnist(True, True)
# clear_ram()

# start_cifa10(True, True)
# clear_ram()

# start_cifar100(True, True)
# clear_ram()