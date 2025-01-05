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
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
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
    # state_dict_ndarrays = [v.cpu().numpy() for v in model.state_dict().values()]
    # parameters = ndarrays_to_parameters(state_dict_ndarrays)
    # return state_dict,state_dict_ndarrays,parameters



def isWeightCorrupted():
    corrected_checksum = requests.get(f'{blockchainApiPrefix}/checksum/weights/{federated_cid}/{round}')
    corrected_checksum = corrected_checksum.text
    
    checksum = calculate_hash()
    
    if checksum != corrected_checksum:
        return True
    return False



def isPoisoned():
    # flattened_weights_server = np.concatenate([
    #     param.detach().cpu().numpy().ravel() for param in server_model.parameters()
    # ])
    # flattened_weights_client = np.concatenate([
    #     param.detach().cpu().numpy().ravel() for param in client_model.parameters()
    # ])
    # diff_nom = np.linalg.norm(flattened_weights_server - flattened_weights_client)#se queste distanca aument troppop cliente avvelenato
    # Utils.printLog(f"client {federated_cid} round {round} dif_nomr {diff_nom}")
    
    
    
    level_counter = 0
    poisoned_level_counter = 0
    for (server_param_name,server_param_value), (client_param_name,client_param_value) in zip(server_model.state_dict().items(), client_model.state_dict().items()):
        if 'bias' in server_param_name:
            pass
        else:
            level_counter = level_counter + 1
            server_param_value = np.array(server_param_value)
            client_param_value = np.array(client_param_value)

            server_max_val = np.max(server_param_value)
            server_min_val = np.min(server_param_value)
            server_param_value = (server_param_value - server_min_val) / (server_max_val - server_min_val)
            
            client_max_val = np.max(client_param_value)
            client_min_val = np.min(client_param_value)
            client_param_value = (client_param_value - client_min_val) / (client_max_val - client_min_val)
            
            # if (((np.abs(server_param_value - client_param_value)) * 100) > percentage_accept_threshold).any():
            #     return True
            if (((np.abs(server_param_value - client_param_value)) * 100) > percentage_accept_threshold).any():
                poisoned_level_counter = poisoned_level_counter + 1 
    
                
    if poisoned_level_counter > (0.30 * level_counter):
        return True

    return False
    

if __name__ == '__main__':
    # istantiate model and load state dict on model
    blockchain_credential = blockchainPrivateKeys[-1]
    
    round = int(sys.argv[3])
    federated_cid = int(sys.argv[4]) 
    
    # percentage_accept_threshold = 100 - (5 * (round - 1))
    percentage = 80 - (100 / int(sys.argv[5])  * (round - 1))
    percentage_accept_threshold = percentage if percentage >= 25 else 25
    
    
    Utils.printLog(f'starting client {federated_cid} check at round {round}')
    
    server_model = Utils.get_model(str(sys.argv[1]), int(sys.argv[2]))
    server_weight_path = f"./data/clientParameters/node/server_round{round - 1}_parameters.pth"
    load_client_model(server_model,server_weight_path)

    client_model = Utils.get_model(str(sys.argv[1]), int(sys.argv[2]))
    client_weight_path = f"./data/clientParameters/node/client{federated_cid}_round{round}_parameters.pth"
    
    client_already_detected = requests.get(f'{blockchainApiPrefix}/poisoners/{federated_cid}')
    if client_already_detected.text == 'true':
        pass
    else:
        if not isWeightCorrupted():
            load_client_model(client_model,client_weight_path)
            if(isPoisoned()):
                Utils.printLog(f'by blockchain script cid {federated_cid} result poisoner at round {round} with threshold {percentage_accept_threshold}%')
                requests.post(f'{blockchainApiPrefix}/write/blacklist/{federated_cid}',
                    json={'blockchainCredential': blockchain_credential})
        else:
            Utils.printLog(f'by blockchain script cid {federated_cid} result poisoner at round {round} by corrupted weight hash')
            requests.post(f'{blockchainApiPrefix}/write/blacklist/{federated_cid}',
                json={'blockchainCredential': blockchain_credential})
    
    Utils.printLog(f'finish client {federated_cid} check at round {round}')

    
        
    
