import numpy as np

CLIENTS_NUM = 10
ROUNDS_NUM = 1
POISONERS_CLIENTS_CID =  np.random.randint(0, CLIENTS_NUM - 1, round((CLIENTS_NUM * 20) / 100))
POISONERS_CLIENTS = 2
