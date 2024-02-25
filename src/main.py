import Server
import Model

from keras.datasets import mnist
from keras.datasets import cifar10
from keras.datasets import cifar100

from flwr.simulation.ray_transport.utils import enable_tf_gpu_growth

enable_tf_gpu_growth()

# cifar10
Model.Model.set_data(cifar10, 10, (4, 4), (32, 32, 3))
server = Server.Server()
server.start_simulation()
