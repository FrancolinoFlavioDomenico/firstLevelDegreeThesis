import sys

import scipy.stats
from src.utils.globalVariable import blockchainApiPrefix,blockchainPrivateKeys
from src.utils.Utils import Utils
import requests
import torch
import warnings
import hashlib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from flwr.common import (
    ndarrays_to_parameters,
)
import scipy
warnings.filterwarnings('ignore')


# Define a function to calculate the SHA-256 hash of a file.
def calculate_hash():
   md5 = hashlib.md5()
   with open(client_weight_path, "rb") as file:
       while True:
           data = file.read(65536)  # Read the file in 64KB chunks.
           if not data:
               break
           md5.update(data)
   return md5.hexdigest()

def load_client_model(model,path):
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)
    state_dict_ndarrays = [v.cpu().numpy() for v in model.state_dict().values()]
    parameters = ndarrays_to_parameters(state_dict_ndarrays)
    return parameters



def isWeightCorrupted():
    corrected_checksum = requests.get(f'{blockchainApiPrefix}/checksum/weights/{federated_cid}/{round}')
    corrected_checksum = corrected_checksum.text
    
    checksum = calculate_hash()
    
    if checksum != corrected_checksum:
        return True
    return False

def flatten_model_weights(model):
    """
    Flattens all the weights of a PyTorch model into a single NumPy array.
    Args:
        model (torch.nn.Module): The PyTorch model to process.
    Returns:
        np.ndarray: A flattened NumPy array containing all weights.
    """
    # Collect all parameters as numpy arrays and flatten them
    flattened_weights = np.concatenate([
        param.detach().cpu().numpy().ravel() for param in model.parameters()
    ])
    return flattened_weights
#param to param
def isPoisoned():
    
    server_param = flatten_model_weights(server_model)
    client_param = flatten_model_weights(client_model)
    diff_nom = np.linalg.norm(server_param - client_param)#se queste distanca aument troppop cliente avvelenato
    Utils.printLog(f"client {federated_cid} round {round} dif_nomr {diff_nom}")

        
    return False

if __name__ == '__main__':
    # istantiate model and load state dict on model
    blockchain_credential = blockchainPrivateKeys[-1]
    
    round = int(sys.argv[3])
    federated_cid = int(sys.argv[4]) 
    
    server_model = Utils.get_model(str(sys.argv[1]), int(sys.argv[2]))
    server_weight_path = f"./data/clientParameters/node/client{0}_round{round - 1}_parameters.pth"
    server_parameters = load_client_model(server_model,server_weight_path)
    
    client_model = Utils.get_model(str(sys.argv[1]), int(sys.argv[2]))
    client_weight_path = f"./data/clientParameters/node/client{federated_cid}_round{round}_parameters.pth"
    
    if not isWeightCorrupted():
        client_parameters = load_client_model(client_model,client_weight_path)
        if(isPoisoned()):
            Utils.printLog(f'by blockchain script cid {federated_cid} result poisoner')
            requests.post(f'{blockchainApiPrefix}/write/blacklist/{federated_cid}',
                json={'blockchainCredential': blockchain_credential})
    else:
        requests.post(f'{blockchainApiPrefix}/write/blacklist/{federated_cid}',
            json={'blockchainCredential': blockchain_credential})
    
        
    
