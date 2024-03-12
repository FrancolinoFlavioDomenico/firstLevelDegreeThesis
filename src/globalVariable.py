import numpy as np

CLIENTS_NUM = 10
ROUNDS_NUM = 5
POISONERS_CLIENTS_CID =  np.random.randint(0, CLIENTS_NUM, round((CLIENTS_NUM * 60) / 100))
