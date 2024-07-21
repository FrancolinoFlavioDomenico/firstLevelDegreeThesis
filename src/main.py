# import sys
# import tensorflow as tf
# from flwr.simulation.ray_transport.utils import enable_tf_gpu_growth
# import flwr as fl
# import logging
# import time
# import gc
#
# import Utils as mf
# import Server
# import globalVariable as gv
#
# DEFAULT_FORMATTER = logging.Formatter(
# "%(levelname)s %(name)s %(asctime)s | %(filename)s:%(lineno)d | %(message)s"
# )
# fl.common.logger.configure(identifier="executionLog", filename="log.txt")
# # enable_tf_gpu_growth()
#
# #####################################
# #
# #      simulation start function
# #
# #####################################
#
# # mnist
# def start_mnist(poisoning=False,blockchain=False):
#     gv.printLog(f"starting mnist dataset {'poisoned' if poisoning else ''}")
#     mnist_model_conf = mf.Utils('mnist', tf.keras.datasets.mnist, 10, (3, 3), (28, 28, 1), poisoning, blockchain)
#     server_mnist = Server.Server(mnist_model_conf)
#     server_mnist.start_simulation()
#     del server_mnist
#     del mnist_model_conf
#     gv.printLog(f"finish mnist {'poisoned' if poisoning else ''}")
#
#
# # cifar10
# def start_cifa10(poisoning=False,blockchain=False):
#     gv.printLog(f"starting cifar10 dataset {'poisoned' if poisoning else ''}")
#     cifar10_model_conf = mf.Utils('cifar10', tf.keras.datasets.cifar10, 10, (3, 3), (32, 32, 3), poisoning, blockchain)
#     server_cifar10 = Server.Server(cifar10_model_conf)
#     server_cifar10.start_simulation()
#     del server_cifar10
#     del cifar10_model_conf
#     gv.printLog(f"finish cifar10 {'poisoned' if poisoning else ''}")
#
#
# # cifar100
# def start_cifar100(poisoning=False,blockchain=False):
#     gv.printLog(f"starting cifar100 dataset {'poisoned' if poisoning else ''}")
#     cifar100_model_conf = mf.Utils('cifar100', tf.keras.datasets.cifar100, 100, (3, 3), (32, 32, 3), poisoning, blockchain)
#     server_cifar100 = Server.Server(cifar100_model_conf)
#     server_cifar100.start_simulation()
#     del server_cifar100
#     del cifar100_model_conf
#     gv.printLog(f"finish cifar100 {'poisoned' if poisoning else ''}")
#
#
# def clear_ram():
#     time.sleep(1)
#     gc.collect()
#     time.sleep(1)
#
#
# #####################################
# #
# #       no poisoning
# #
# #####################################
# # start_mnist(False)
# # clear_ram()
#
# # start_cifa10(False)
# # clear_ram()
#
# start_cifar100(False)
# clear_ram()
#
#
# #####################################
# #
# #        poisoning
# #
# #####################################
# # start_mnist(True)
# # clear_ram()
#
# # start_cifa10(True)
# # clear_ram()
#
# # start_cifar100(True)
# # clear_ram()
#
#
# #####################################
# #
# #       no poisoning blockchian
# #
# #####################################
# # start_mnist(False,True)
# # clear_ram()
#
# # start_cifa10(False, True)
# # clear_ram()
#
# # start_cifar100(False, True)
# # clear_ram()
#
#
# #####################################
# #
# #       poisoning blockchian
# #
# #####################################
# # start_mnist(True, True)
# # clear_ram()
#
# # start_cifa10(True, True)
# # clear_ram()
#
# # start_cifar100(True, True)
# # clear_ram()

import time
import Utils
import Server
import FlowerClient
import multiprocessing as mp
import subprocess
import flwr as fl
import threading
import logging

import threading as th

DEFAULT_FORMATTER = logging.Formatter(
    "%(levelname)s %(name)s %(asctime)s | %(filename)s:%(lineno)d | %(message)s"
)
fl.common.logger.configure(identifier="executionLog", filename="log.txt")




def start_server():
    server = Server.Server(utils)
    # subporcess
    # server.start_server()
    # simulation
    server.start_simulation()


def start_client(cid):
    print(cid)
    client = FlowerClient.FlowerClient(utils, cid)
    client.start_client()


def client_test():
    import clientTest
    clientTest = clientTest.FlowerClient(utils,0)
    clientTest.test()


if __name__ == "__main__":
    # utils = Utils.Utils('cifar100', 100, False, False)
    # start_server()
    # time.sleep(15)
    # utils = Utils.Utils('cifar100', 100, True, False)
    # start_server()
    
    # time.sleep(30)
    
    # utils = Utils.Utils('cifar10', 10, False, False)
    # start_server()
    # time.sleep(15)
    # utils = Utils.Utils('cifar10', 10, True, False)
    # start_server()
    
    # time.sleep(30)
    
    # utils = Utils.Utils('mnist', 10, False, False)
    # start_server()
    # time.sleep(15)
    # utils = Utils.Utils('mnist', 10, True, False)
    # start_server()
    
    utils = Utils.Utils('mnist', 10, False,True)
    time.sleep(15)
    start_server()
    
    
    
    print("python started")
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
    # # tescode
    # # client_test()

    # # simulation
    # # utils = Utils.Utils('mnist', 10, (3, 3), (28, 28, 1), False, False)
    # # utils = Utils.Utils('cifar10', 10, (3, 3), (32, 32, 3), True, False)
    # utils = Utils.Utils('cifar100', 100, (3, 3), (32, 32, 3), False, False)
    # start_server()
    
    # # time.sleep(15)
    
    # # utils = Utils.Utils('mnist', 10, (3, 3), (28, 28, 1), True, False)
    # # start_server()
    

    # # subprocess
    # # serverThread = mp.Process(target=start_server)
    # # serverThread.start()
    # # time.sleep(15)
    # # for i in range(Utils.Utils.CLIENTS_NUM):
    # #     client_thread  = th.Thread(target=start_client,args=[i])
    # #     client_thread.start()
    # #     client_thread.join
    # # serverThread.join()

