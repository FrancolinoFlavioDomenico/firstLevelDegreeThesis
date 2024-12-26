import sys
from src.utils.globalVariable import blockchainApiPrefix,blockchainPrivateKeys
from src.utils.Utils import Utils
import requests
import torch
import warnings
import hashlib
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

def load_weights_to_check(path):
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)

def isWeightCorrupted(path,correctedChecksum):
    checksum = calculate_hash(path)
    if checksum != correctedChecksum:
        return True
    return False

def isPoisoned():
    # TODO implement weight check
    # load last round server weight
    # check current client weight
    #  if weigth < avgWeitg of previous round
    #           add clientCid to blacklist
    #           write weigt on blockchain
    #           write blacklist on blockchain
    
    return False


if __name__ == '__main__':
    blockchain_credential = blockchainPrivateKeys[-1]
    
    model = Utils.get_model(str(sys.argv[1]), int(sys.argv[2]))
    round = int(sys.argv[3])
    federated_cid = int(sys.argv[4]) 
    weightPath = f"./data/clientParameters/node/client{federated_cid}_round{round}_parameters.pth"
    correctedChecksum = requests.get(f'{blockchainApiPrefix}checksum/weights/{federated_cid}/{round}')
    correctedChecksum = correctedChecksum.text
    
    if not isWeightCorrupted(weightPath,correctedChecksum):
        load_weights_to_check(weightPath)
    else:
        requests.post(f'{blockchainApiPrefix}write/blacklist/{federated_cid}',
            json={'blockchainCredential': blockchain_credential})
        
    
    if(isPoisoned()):
        requests.post(f'{blockchainApiPrefix}write/blacklist/{federated_cid}',
            json={'blockchainCredential': blockchain_credential})
