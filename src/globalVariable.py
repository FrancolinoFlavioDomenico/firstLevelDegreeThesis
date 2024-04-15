import numpy as np
import logging


CLIENTS_NUM = 10
ROUNDS_NUM = 5
POISONERS_CLIENTS_CID =  np.random.randint(0, CLIENTS_NUM, round((CLIENTS_NUM * 30) / 100))
POISONERS_CLIENTS = 2
partitions_index_list = np.arange(0,CLIENTS_NUM)

    
def printLog(msg):
    print(msg)
    logging.log(logging.INFO,msg)
