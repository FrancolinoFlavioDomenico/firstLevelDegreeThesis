from federation import Server
import flwr as fl
import logging
import time
from utils import Configuration
import subprocess
import sys
import psutil
from src.utils.Utils import Utils

DEFAULT_FORMATTER = logging.Formatter(
    "%(levelname)s %(name)s %(asctime)s | %(filename)s:%(lineno)d | %(message)s"
)
fl.common.logger.configure(identifier="executionLog", filename="log/federation_log.txt")

def start_server(configuration):
    server = Server.Server(configuration)
    server.start_simulation()


def start(dataset_name,classes_number,poisoning,blockchian):
    configuration = Configuration.Configuration(dataset_name, classes_number, poisoning, blockchian)
    time.sleep(5) # waiting for partition writing
    start_server(configuration)
    
def start_blockchain():
    subprocess.run("start npm run start-blockchain",shell=True) 
    time.sleep(25) # waiting blockchain starting
    subprocess.run("start npm run start-server",shell=True) 
    time.sleep(5) # waiting server starting
    
    
def kill_process_by_name(process_name):
    for proc in psutil.process_iter():
        if process_name.lower() in proc.name().lower():
            proc.kill()
            
def run_experiment(dataset_name, classes_number):
    Utils.printLog(f'Starting pure scenery on {dataset_name}')
    start(dataset_name, classes_number, False, False)
    time.sleep(5) # waiting garbage collection
    
    Utils.printLog(f'\n\n\nStarting poisoning scenery on {dataset_name}')
    start(dataset_name, classes_number, True, False)
    time.sleep(5) # waiting garbage collection
    
    Utils.printLog(f'\n\n\nStarting poisoning with mitigation system scenery on {dataset_name}')
    start_blockchain()
    start(dataset_name, classes_number, True, True)
    
    kill_process_by_name('node')
    Utils.printLog(f'\n\n\n\n\n')


if __name__ == "__main__":
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
        if dataset_name.lower() in ['mnist', 'cifar10', 'cifar100']:
            classes_number = 10 if dataset_name != 'cifar100' else 100
            run_experiment(dataset_name, classes_number)
        else:
            print("Invalid dataset name. Please use 'mnist', 'cifar10', or 'cifar100'.")
    else:
        run_experiment('mnist', 10)



    
    
    
    