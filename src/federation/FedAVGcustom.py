import flwr as fl

from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

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
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from src.utils.globalVariable import blockchainApiPrefix,blockchainPrivateKeys
import requests
import torch
from src.utils.Utils import Utils



class FedAVGcustom(fl.server.strategy.FedAvg):    
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        #TODO (review code)remove poisoned result and then call overided method
        if server_round >= 2:
            # blacklist  = requests.get('blacklist')
            for client_proxy, fit_res in results:
                is_client_poisoner = requests.get(f'{blockchainApiPrefix}/poisoners/{client_proxy.cid}')
                if is_client_poisoner.text == 'true':
                    Utils.printLog(f"Client {client_proxy.cid} is a poisoner...removing it")
                    results.remove((client_proxy,fit_res))
            
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round,results,failures)
        path = f"./data/clientParameters/python/client{0}_round{server_round}_parameters.pth"
        torch.save(parameters_aggregated,path)
        with open(path, 'rb') as f:
            requests.post(f'{blockchainApiPrefix}/write/weights/{0}/{server_round}',
                              data={'blockchainCredential': blockchainPrivateKeys[0]},
                                files={"weights": f})
        return parameters_aggregated, metrics_aggregated