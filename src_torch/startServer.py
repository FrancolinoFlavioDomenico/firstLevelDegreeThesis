from Server import Server
from Utils import Utils

utils = Utils('cifar100', 100, (3, 3), (32, 32, 3), False, False)

server = Server(utils)
server.start_server()
