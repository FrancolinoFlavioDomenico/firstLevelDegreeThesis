from FlowerClient import FlowerClient
from Utils import Utils
utils = Utils('cifar100',100, (3, 3), (32, 32, 3), False, False)

client = FlowerClient(utils,0)
client.start_client()