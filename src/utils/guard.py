import sys
import time
import torch

if __name__ == '__main__':
    weightAddress = str(sys.argv[1])# Takes number from command line argument
    round = int(sys.argv[2])# Takes number from command line argument
    federatedCid = int(sys.argv[3]) 
    #TODO implement weight check