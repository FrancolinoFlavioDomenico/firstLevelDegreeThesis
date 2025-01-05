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


#param concatenated using np.abs
# def isPoisoned():
#     flatten_server_param = np.empty(1)
#     flatten_client_param = np.empty(1)
#     for server_param, client_param in zip(server_model.parameters(), client_model.parameters()):
#         flatten_server_param = np.concatenate((flatten_server_param,np.array(server_param.data).flatten())) 
#         flatten_client_param = np.concatenate((flatten_client_param,np.array(client_param.data).flatten())) 
#     flatten_server_param = flatten_server_param.flatten()
#     flatten_client_param = flatten_client_param.flatten()

#     flatten_server_param_mean = np.mean(flatten_server_param)
#     flatten_client_param_mean = np.mean(flatten_client_param)

#     flatten_server_param_std = np.std(flatten_server_param)
#     flatten_client_param_std = np.std(flatten_client_param)
    
#     lower_limit = np.abs(flatten_server_param_mean) - np.abs(20 * flatten_server_param_std)
#     upper_limit = np.abs(flatten_server_param_mean) + np.abs(20 * flatten_server_param_std)
#     if (np.abs(flatten_client_param) < np.abs(lower_limit)).any() or (np.abs(flatten_client_param) > np.abs(upper_limit)).any():
#         Utils.printLog(f"\nlower_limit {lower_limit}")
#         Utils.printLog(f"\nupper_limit {upper_limit}")
#         # Utils.printLog(f"\nflatten_server_param_std {flatten_server_param_std}")
#         Utils.printLog(f"\nflatten_client_param_ {flatten_client_param}")
#         Utils.printLog(f"\nflatten_server_param_ {flatten_server_param}")
#         Utils.printLog(f"\nflatten_server_param_mean {flatten_server_param_mean}")
#         Utils.printLog(f"\nflatten_client_param_mean {flatten_client_param_mean}")
#         return True
        
#     return False

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
    
    
    # arr_server = np.array(1)
    # client_server = np.array(1)
    # for server_param, client_param in zip(server_model.parameters(), client_model.parameters()):
    #     arr = np.append()
    #     flatten_server_param = np.array(server_param.data).flatten() 
    #     flatten_client_param = np.array(client_param.data).flatten() 


    #     diff_nom = np.linalg.norm(flatten_server_param - flatten_client_param)#se queste distanca aument troppop cliente avvelenato
    #     Utils.printLog(f"client {federated_cid} round {round} dif_nomr {diff_nom}")
        
        # diff_nom = np.linalg.norm(flatten_server_param - flatten_client_param, 'fro')#se queste distanca aument troppop cliente avvelenato
        # Utils.printLog(f"client {federated_cid} round {round} dif_nomr {diff_nom}")
        
        # if diff_nom > 5:
        #     return True

        
    return False

# def isPoisoned():
#     server_state_dict = server_model.state_dict()
#     client_state_dict = client_model.state_dict()
#     for name, tensor1 in server_state_dict.items():
#         tensor2 = client_state_dict[name]
#         tensor1_mean = torch.mean(tensor1)
#         tensor2_mean = torch.mean(tensor2)
#         tensor1_std= torch.std(tensor1)
#         tensor2_std= torch.std(tensor2)
#         lower_limit = tensor1_mean - (3 * tensor1_std)
#         upper_limit = tensor1_mean + (3 * tensor1_std)
#         if (np.array(tensor2) < np.array(lower_limit)).any()  or (np.array(tensor2) > np.array(upper_limit)).any() :
#             Utils.printLog(f"\nlower_limit {lower_limit}")
#             Utils.printLog(f"\nupper_limit {upper_limit}")
#             # Utils.printLog(f"\nflatten_server_param_std {flatten_server_param_std}")
#             Utils.printLog(f"\nflatten_client_param_ {tensor2}")
#             Utils.printLog(f"\nflatten_server_param_ {tensor1}")
#             Utils.printLog(f"\nflatten_server_param_mean {tensor1_mean}")
#             Utils.printLog(f"\nflatten_client_param_mean {tensor2_mean}")
#             return True
#     return False

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
    
    # if not isWeightCorrupted():
    client_parameters = load_client_model(client_model,client_weight_path)
    if(isPoisoned()):
        Utils.printLog(f'by blockchain script cid {federated_cid} result poisoner')
        requests.post(f'{blockchainApiPrefix}/write/blacklist/{federated_cid}',
            json={'blockchainCredential': blockchain_credential})
    # else:
    #     requests.post(f'{blockchainApiPrefix}/write/blacklist/{federated_cid}',
    #         json={'blockchainCredential': blockchain_credential})
    
        
    
