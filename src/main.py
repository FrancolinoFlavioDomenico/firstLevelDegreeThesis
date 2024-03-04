import Server
import ModelConf as mf

from keras.datasets import mnist
from keras.datasets import cifar10
from keras.datasets import cifar100

from flwr.simulation.ray_transport.utils import enable_tf_gpu_growth

import time

enable_tf_gpu_growth()

# cifar10_model_conf = mf.ModelConf('cifar10', cifar100, 10, (4, 4), (32, 32, 3))



print("starting cifar10 dataset")
# cifar10
cifar10_model_conf = mf.ModelConf('cifar10', cifar10, 10, (4, 4), (32, 32, 3))
server_cifar10 = Server.Server(cifar10_model_conf)
server_cifar10.start_simulation()
del server_cifar10
del cifar10_model_conf
print("finish cifar10")

time.sleep(10)
print("starting mnist dataset")

# # mnist
# mnist_model_conf = mf.ModelConf('mnist', mnist, 10, (3, 3), (28, 28, 1))
# server_mnist = Server.Server(mnist_model_conf)
# server_mnist.start_simulation()
# del server_mnist
# del mnist_model_conf
# print("finish mnist")
#
# time.sleep(10)
# print("starting cifar100 dataset")
#
# # cifar100
# cifar100_model_conf = mf.ModelConf('cifar100', cifar100, 100, (4, 4), (32, 32, 3))
# server_cifar100 = Server.Server(cifar100_model_conf)
# server_cifar100.start_simulation()
# del server_cifar100
# del cifar100_model_conf
# print("finish cifar100")
