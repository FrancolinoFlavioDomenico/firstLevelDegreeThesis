     
from server import Server
import model as m



from flwr.simulation.ray_transport.utils import enable_tf_gpu_growth

from keras.datasets import mnist
from keras.datasets import cifar10
from keras.datasets import cifar100

# Enable GPU growth in main process
enable_tf_gpu_growth()

#Cifar10
m.Model.setData(cifar10, 10, (4,4), (32, 32, 3))
m.Model.setModel()
server = Server()