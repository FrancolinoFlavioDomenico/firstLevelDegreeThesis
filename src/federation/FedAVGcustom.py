import flwr as fl

from typing import Callable, Dict, List, Optional, Tuple, Union

from flwr.common import (
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy

from src.utils.globalVariable import blockchainApiPrefix,blockchainPrivateKeys
import requests
import torch
import numpy as np
from src.utils.Utils import Utils
from collections import OrderedDict
import time


class FedAVGcustom(fl.server.strategy.FedAvg):    
    def __init__(        
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        inplace: bool = True,
        dataset_name = 'mnist',
        classes_number = 10
        ):
        super().__init__(
                min_fit_clients=min_fit_clients,
                min_evaluate_clients=min_evaluate_clients,
                min_available_clients=min_available_clients,
                evaluate_fn=evaluate_fn,
                fraction_evaluate=fraction_evaluate,
                on_fit_config_fn=on_fit_config_fn
                )
        self.dataset_name = dataset_name
        self.classes_number = classes_number
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        time.sleep(5)
        
        Utils.printLog(f"\n\nstarting custom aggregration for round {server_round}")
        
        tuple_to_remove = []
        if server_round >= 2:
            for client_proxy, fit_res in results:
                is_client_poisoner = requests.get(f'{blockchainApiPrefix}/poisoners/{client_proxy.cid}')
                Utils.printLog(f"Client {client_proxy.cid} is in blockchain blacklist: {is_client_poisoner.text}")
                if is_client_poisoner.text == 'true':
                    Utils.printLog(f"Client {client_proxy.cid} is a poisoner...removing it")
                    tuple_to_remove.append((client_proxy,fit_res))
        
            results =  [t1 for t1 in results if not any(t1 == t2 for t2 in tuple_to_remove)]
            
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round,results,failures)
        
        aggregated_ndarrays: List[np.ndarray] = parameters_to_ndarrays(parameters_aggregated)
        model = Utils.get_model(self.dataset_name,self.classes_number)
        
        params_dict = zip(model.state_dict().keys(), aggregated_ndarrays)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
            
        path = f"./data/clientParameters/python/server_round{server_round}_parameters.pth"
        torch.save(model.state_dict(),path)
        with open(path, 'rb') as f:
            requests.post(f'{blockchainApiPrefix}/write/weights/server/{server_round}',
                            data={'blockchainCredential': blockchainPrivateKeys[-1]},
                                files={"weights": f})
            
        Utils.printLog(f"\n\nfinish custom aggregration for round {server_round}")
        return parameters_aggregated, metrics_aggregated