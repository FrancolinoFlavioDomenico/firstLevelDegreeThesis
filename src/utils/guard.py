import sys

from src.utils.globalVariable import blockchainApiPrefix,blockchainPrivateKeys
from src.utils.Utils import Utils
import requests
import torch
import warnings
import hashlib
import numpy as np

warnings.filterwarnings('ignore')


# Define a function to calculate the SHA-256 hash of a file.
def calculate_hash(path):
   md5 = hashlib.md5()
   with open(path, "rb") as file:
       while True:
           data = file.read(65536)  # Read the file in 64KB chunks.
           if not data:
               break
           md5.update(data)
   return md5.hexdigest()

def load_client_model(model,path):
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)

def isWeightCorrupted(federated_cid,round,path):
    corrected_checksum = requests.get(f'{blockchainApiPrefix}/checksum/weights/{federated_cid}/{round}')
    corrected_checksum = corrected_checksum.text
    
    checksum = calculate_hash(path)
    
    if checksum != corrected_checksum:
        return True
    return False



def isPoisoned():      
    # norm difference method
    flattened_weights_server = np.empty(1)
    flattened_weights_client = np.empty(1)
    for (server_param_name,server_param_value), (client_param_name,client_param_value) in zip(server_model.state_dict().items(), client_model.state_dict().items()):
        if 'bias' in server_param_name:
            pass
        else:
            flattened_weights_server = np.concatenate((flattened_weights_server,np.array(server_param_value).flatten()))
            flattened_weights_client = np.concatenate((flattened_weights_client,np.array(client_param_value).flatten()))
    
    server_nom = np.linalg.norm(flattened_weights_server)
    client_nom = np.linalg.norm(flattened_weights_client)
    
    diff_nom = client_nom  - server_nom
    difference_percentage = (diff_nom / server_nom)  * 100
    Utils.printLog(f"\n client: {federated_cid}\n round: {round} \n dif_nomr: {diff_nom}\n percentage: {difference_percentage}\n current threshold: {percentage_accept_threshold}")
    if difference_percentage > percentage_accept_threshold or difference_percentage < (0 - percentage_accept_threshold):
        return True
    
    return False
    

if __name__ == '__main__':
    dataset_name = str(sys.argv[1])
    classes_count = int(sys.argv[2])
    total_round_count = int(sys.argv[3])
    total_client_count = int(sys.argv[4])
    round = int(sys.argv[5])
    federated_cid = int(sys.argv[6]) 
    
    blockchain_credential = blockchainPrivateKeys[total_client_count]
    
    percentage_tollerance = 10 - (10 / total_round_count * (round - 1))
    percentage_tollerance = percentage_tollerance if percentage_tollerance >= 0 else 0
    percentage_to_remove = ((100) - ((100 + percentage_tollerance) / total_round_count  * (round - 1))) + percentage_tollerance
    if round == 2:#first check
        percentage_to_remove = 100 + percentage_tollerance
    percentage_accept_threshold = percentage_to_remove if percentage_to_remove >= 30 else 30
    
    server_model = Utils.get_model(dataset_name, classes_count)
    server_weight_path = f"./data/clientParameters/node/server_round{round - 1}_parameters.pth"
    load_client_model(server_model,server_weight_path)

    client_model = Utils.get_model(dataset_name, classes_count)
    client_weight_path = f"./data/clientParameters/node/client{federated_cid}_round{round}_parameters.pth"
    
    client_already_detected = requests.get(f'{blockchainApiPrefix}/poisoners/{federated_cid}')
    if client_already_detected.text == 'true':
        pass
    else:
        if not isWeightCorrupted(federated_cid,round,client_weight_path):
            load_client_model(client_model,client_weight_path)
            if(isPoisoned()):
                Utils.printLog(f'by blockchain script cid {federated_cid} result poisoner at round {round} with threshold {percentage_accept_threshold}%')
                requests.post(f'{blockchainApiPrefix}/write/blacklist/{federated_cid}',
                    json={'blockchainCredential': blockchain_credential})
        else:
            Utils.printLog(f'by blockchain script cid {federated_cid} result poisoner at round {round} by corrupted weight hash')
            requests.post(f'{blockchainApiPrefix}/write/blacklist/{federated_cid}',
                json={'blockchainCredential': blockchain_credential})
