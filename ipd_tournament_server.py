from collections import defaultdict
import random
from logging import INFO
from flwr.common.logger import log
from flwr.server.strategy import Strategy
from flwr.server import Server
import concurrent.futures

from typing import Optional, Union

from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    GetPropertiesIns,
    GetPropertiesRes,
    Code,
    Properties
)
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.server import fit_clients


FitResultsAndFailures = tuple[
    list[tuple[ClientProxy, FitRes]],
    list[Union[tuple[ClientProxy, FitRes], BaseException]],
]

class Ipd_ClientManager(SimpleClientManager):
     def __init__(self) -> None:
         super().__init__()
         log(INFO, "Starting Ipd client manager")
         self.matchmaking_dict = dict()


class Ipd_TournamentServer(Server):
    def __init__(
        self,
        *,
        client_manager: Ipd_ClientManager,
        strategy: Optional[Strategy] = None,
    ) -> None:
        super().__init__(client_manager=client_manager, strategy=strategy)
        self.matchmaking_dict = dict()
        
    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        tuple[Optional[Parameters], dict[str, Scalar], FitResultsAndFailures]
    ]:
        """Perform a single round of federated averaging."""
        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )
        
        if server_round > 1:
            self.ipd_matchmaking(client_instructions, max_workers=self.max_workers, timeout=timeout, server_round=server_round)

        if not client_instructions:
            log(INFO, "configure_fit: no clients selected, cancel")
            return None
        log(
            INFO,
            "configure_fit: strategy sampled %s clients (out of %s)",
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `fit` results from all clients participating in this round
        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
            group_id=server_round,
        )
        log(
            INFO,
            "aggregate_fit: received %s results and %s failures",
            len(results),
            len(failures),
        )
        
        self.resolve_ipd_matchmaking(results)

        # Aggregate training results
        aggregated_result: tuple[
            Optional[Parameters],
            dict[str, Scalar],
        ] = self.strategy.aggregate_fit(server_round, results, failures)

        parameters_aggregated, metrics_aggregated = aggregated_result
        return parameters_aggregated, metrics_aggregated, (results, failures)
    
    def ipd_matchmaking(self, client_instructions, max_workers, timeout, server_round):
        # collect clients 
        participating_client_lst = get_properties_async(client_instructions=client_instructions,
                                                        max_workers=max_workers,
                                                        timeout=timeout,
                                                        group_id=server_round)
        '''
        for idx, client in enumerate(client_instructions):
            conf = {'client_id': 0, 'strategy': ''}
            props = GetPropertiesIns(config=conf)
            client_prob_dict = client[0].get_properties(props, timeout=timeout, group_id=server_round)
            participating_client_lst.append((idx, client_prob_dict.properties))
        '''
        
        
        # shuffle list and pop last two
        random.shuffle(participating_client_lst)
        
        while len(participating_client_lst) > 1:
            # check client participation in previous round
            player_1 = participating_client_lst.pop()
            player_2 = participating_client_lst.pop()
            # create Key
            id_p1 = int(player_1[1]["client_id"])
            id_p2 = int(player_2[1]["client_id"])
            # sort
            sorted_x, sorted_y = sorted([id_p1, id_p2])
            # calc unique match hash
            hash_key = str(cantor_pairing(sorted_x, sorted_y))
            # obtain history
            if hash_key in self.matchmaking_dict:
                matchup_1, matchup_2 = self.matchmaking_dict[hash_key]
                # integrate matchups in FitRes of client A
                {"ipd_history_plays": 0, "ipd_history_coplays": 0}
                # fix flip by sort operation
                if sorted_x == int(player_1[1]["client_id"]):
                    client_instructions[player_1[0]][1].config["ipd_history_plays"] = matchup_1
                    client_instructions[player_1[0]][1].config["ipd_history_coplays"] = matchup_2
                    # integrate matchups in FitRes of client B
                    client_instructions[player_2[0]][1].config["ipd_history_plays"] = matchup_2
                    client_instructions[player_2[0]][1].config["ipd_history_coplays"] = matchup_1
                else:
                    client_instructions[player_1[0]][1].config["ipd_history_plays"] = matchup_2
                    client_instructions[player_1[0]][1].config["ipd_history_coplays"] = matchup_1
                    # integrate matchups in FitRes of client B
                    client_instructions[player_2[0]][1].config["ipd_history_plays"] = matchup_1
                    client_instructions[player_2[0]][1].config["ipd_history_coplays"] = matchup_2
                # attach match_id
            client_instructions[player_1[0]][1].config["match_id"] = hash_key
            client_instructions[player_2[0]][1].config["match_id"] = hash_key
            
            
          
    def resolve_ipd_matchmaking(self, results):
        log(INFO, "Running IPD matchmaking resolve")
        if not results:
            return None, {}
        
        # Collect results
        matchup_results = [
            (fit_res.metrics, fit_res.num_examples)
            for _, fit_res in results
        ]
        
        # Group dictionaries by match_id
        grouped_data = defaultdict(list)
        for metrics, num_examples in matchup_results:
            grouped_data[metrics["match_id"]].append((metrics, num_examples))
        
        # entry -> (dict[] metrics, int num), metrics -> [client_id, match_id]
        # process double-entries
        for match_id, entries in grouped_data.items():
            if len(entries) > 1: # found more than one match_id entry
                #print(f"Duplicate match_id {match_id} found with entries:")
                log(INFO, "Duplicate match_ids found")
                for i in range(len(entries) - 1):
                    # extract results
                    metrics_1, num_examples_1 = entries[i]
                    metrics_2, num_examples_2 = entries[i + 1]
                    # fetch server history by match_id
                    match_id = metrics_1["match_id"]
                    c1_id = int(metrics_1["client_id"])
                    c2_id = int(metrics_2["client_id"])
                    if match_id in self.matchmaking_dict:
                         # update entries with match result
                        history_c1, history_c2 = self.matchmaking_dict[match_id]
                    else:
                        # create new entry
                        history_c1 = 0
                        history_c2 = 0
                        
                    if c1_id < c2_id:
                        action_c1 = (True if num_examples_1 > 0 else False)
                        action_c2 = (True if num_examples_2 > 0 else False)
                    else:
                        action_c1 = (True if num_examples_2 > 0 else False)
                        action_c2 = (True if num_examples_1 > 0 else False)
                    history_c1 = append_bool_to_msb(history_c1, action_c1)
                    history_c2 = append_bool_to_msb(history_c2, action_c2)
                    # SCORING HAPPENS HERE!!!!!! score(...)
                    self.matchmaking_dict[match_id] = (history_c1, history_c2)
            else:
                log(INFO, "Single match_id found")

def get_properties(
    client: ClientProxy, ins: GetPropertiesIns, timeout: Optional[float], group_id: int, idx: int
):
    """Refine parameters on a single client."""
    prop_res = client.get_properties(ins, timeout=timeout, group_id=group_id)
    return client, prop_res, idx

def get_properties_async(
        client_instructions: list[tuple[ClientProxy, FitRes]],
        max_workers: Optional[int],
        timeout: Optional[float],
        group_id: int,
    ) -> list:
        """Refine parameters concurrently on all selected clients."""
        #participating_client_lst = list()
        '''
        for idx, client in enumerate(client_instructions):
            conf = {'client_id': 0, 'strategy': ''}
            props = GetPropertiesIns(config=conf)
            client_prob_dict = client[0].get_properties(props, timeout=timeout, group_id=server_round)
            participating_client_lst.append((idx, client_prob_dict.properties))
            '''
        conf = {'client_id': 0, 'strategy': ''}
        props = GetPropertiesIns(config=conf)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            submitted_fs = {
                executor.submit(get_properties, client_proxy, props, timeout, group_id, idx)
                for idx, (client_proxy, _) in enumerate(client_instructions)
            }
            finished_fs, _ = concurrent.futures.wait(
                fs=submitted_fs,
                timeout=None,  # Handled in the respective communication stack
            )

        # Gather results
        results: list[tuple[int, Properties]] = []
        failures: list[Union[tuple[ClientProxy, GetPropertiesRes], BaseException]] = []
        for future in finished_fs:
            _handle_finished_future_after_get_properties(
                future=future, results=results, failures=failures
            )
            
        return results


def _handle_finished_future_after_get_properties(
    future: concurrent.futures.Future,  # type: ignore
    results: list[tuple[int, Properties]],
    failures: list[Union[tuple[ClientProxy, GetPropertiesRes], BaseException]],
) -> None:
    """Convert finished future into either a result or a failure."""
    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        failures.append(failure)
        return

    # Successfully received a result from a client
    result: tuple[ClientProxy, GetPropertiesRes, int] = future.result()
    _, res, idx = result

    # Check result status code
    if res.status.code == Code.OK:
        results.append((idx, res.properties))
        return

    # Not successful, client returned a result where the status code is not OK
    failures.append(result)


def cantor_pairing(x, y):
    """Combine two non-negative integers into a single hash key using Cantor's pairing function."""
    return (x + y) * (x + y + 1) // 2 + y

def reverse_cantor_pairing(z):
    """Retrieve the original pair of numbers (x, y) from the Cantor's pairing function result."""
    # Solve for x and y from the given z (the hash key)
    w = int(((8 * z + 1)**0.5 - 1) // 2)  # Inverse of the quadratic equation
    t = (w * (w + 1)) // 2
    y = z - t
    x = w - y
    return x, y

def append_bool_to_msb(n, new_bool):
    # Find the number of bits in the integer
    num_bits = n.bit_length()
    
    # Shift the integer left by 1 to make space for the new MSB
    n = n << 1
    
    # If the new boolean is True, set the most significant bit to 1
    if new_bool:
        n += 1 << num_bits  # Add 1 at the MSB position
    
    return n
        
        
        

    