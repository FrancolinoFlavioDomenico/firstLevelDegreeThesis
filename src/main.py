from utils import Utils
from federation import Server
import flwr as fl
import logging
import numpy as np


DEFAULT_FORMATTER = logging.Formatter(
    "%(levelname)s %(name)s %(asctime)s | %(filename)s:%(lineno)d | %(message)s"
)
fl.common.logger.configure(identifier="executionLog", filename="log/federation_log.txt")




def start_server():
    server = Server.Server(utils)
    server.start_simulation()
    # TODO remove at and end of project
    # subporcess
    # server.start_server()
    # simulation

# TODO remove at and end of project
# def start_client(cid):
#     print(cid)
#     client = FlowerClient.FlowerClient(utils, cid)
#     client.start_client()


# def client_test():
#     import clientTest
#     clientTest = clientTest.FlowerClient(utils,0)
#     clientTest.test()


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
    # utils = Utils.Utils('mnist', 10, False, False)
    # start_server()

    # utils = Utils.Utils('mnist', 10, True, False)
    # start_server()


    utils = Utils.Utils('cifar10', 10, False, False)
    start_server()

    # utils = Utils.Utils('cifar10', 10, True, False)
    # start_server()


    # utils = Utils.Utils('cifar100', 100, False, False)
    # start_server()

    # utils = Utils.Utils('cifar100', 100, True, False)
    # start_server()

   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
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

