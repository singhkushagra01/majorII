import flwr as fl

from typing import List, Tuple, Union, Optional, Dict
import numpy as np

from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes, Parameters, Scalar

class SaveModelStrategy(fl.server.strategy.FedAdagrad):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        # Call aggregate_fit from base class (FedAdagrad) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Save aggregated_ndarrays
            print(f"Saving round {server_round} aggregated_ndarrays...")
            np.savez(f"FedAdagrad - round-{server_round}-weights.npz", *aggregated_ndarrays)

        return aggregated_parameters, aggregated_metrics

# Define an averaging strategy
#strategy = fl.server.strategy.FedAvg(fraction_fit=0.5)
strategy = SaveModelStrategy()
# Start the server with the defined strategy
fl.server.start_server(
    server_address="127.0.0.1:5000",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy
)
