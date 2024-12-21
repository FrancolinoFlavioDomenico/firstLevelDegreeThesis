from flask import Flask
import sys
from src.utils.globalVariable import blockchainApiPrefix,blockchainPrivateKeys
from src.utils.Utils import Utils
import requests
import torch
import flwr as fl
import io
import warnings

warnings.filterwarnings('ignore')


blockchain_credential = blockchainPrivateKeys[-1]

import hashlib

# Define a function to calculate the SHA-256 hash of a file.
def calculate_hash(file_path):
   md5 = hashlib.md5()
   with open(file_path, "rb") as file:
       while True:
           data = file.read(65536)  # Read the file in 64KB chunks.
           if not data:
               break
           md5.update(data)
   return md5.hexdigest()

def get_weights_to_check(weightsHash,round,federated_cid):
    path = f"./data/clientParameters/node/client{federated_cid}_round{round}_parameters.pth"
    checksum = calculate_hash(path)
    with open("checksum.txt", "w") as file:
        file.write("Checksum python: " + checksum + "\n")
        file.write("Checksum node: " + weightsHash + "\n")
    if checksum != weightsHash:
        print("Checksum mismatch. File may be corrupted or tampered with.")
        return
    state_dict = torch.load(path)
    path = f"test.pth"
    model.load_state_dict(state_dict)
    torch.save(model.state_dict(),path)


def isPoisoned():
    # TODO implement weight check
    #  if weigth < avgWeitg of previous round
    #           add clientCid to blacklist
    #           write weigt on blockchain
    #           write blacklist on blockchain
    
    return False



if __name__ == '__main__':
    model = Utils.get_model(str(sys.argv[4]), int(sys.argv[5]))
    weightsHash = str(sys.argv[1])
    round = int(sys.argv[2])
    federated_cid = int(sys.argv[3]) 
    
    get_weights_to_check(weightsHash,round,federated_cid)
    if(isPoisoned()):
        requests.post(f'{blockchainApiPrefix}write/blacklist/{federated_cid}',
            json={'blockchainCredential': blockchain_credential})



