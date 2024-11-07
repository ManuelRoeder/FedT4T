from flwr.server.strategy import FedAvg
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from flwr.common import (
    FitIns,
    Parameters,
    FitRes,
    Scalar
)

from typing import Optional, Union

class Ipd_TournamentStrategy(FedAvg):
    """Provides a pool of available clients."""
    
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]
    
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        # Calculate the total number of examples used during training
        num_examples_total = sum(num_examples for (_, num_examples) in results)
        # hack to avoid DivisionByZero for all defect rounds
        if num_examples_total > 0:
            return super().aggregate_fit(server_round=server_round,
                                         results=results,
                                         failures=failures)
        else:
            return None, {}
    
