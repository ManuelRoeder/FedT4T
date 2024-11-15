import copy
import random
import util
import axelrod as axl
from axelrod.action import Action
from axelrod.strategies import WinStayLoseShift

from ipd_player import ClientShadowPlayer, ResourceAwareMemOnePlayer, RandomIPDPlayer

NUM_GAMES = 20

random.seed(42)

def test_run(client_player):
    # history list received form server instance
    plays = list()
    coplays = list()
    
    for i in range(NUM_GAMES):
        # init shadow player
        surrogate = ClientShadowPlayer()
        surrogate.set_seed(42)
        
        # reset client strategy
        client_player.reset()
        client_player.set_seed(42)
        
        # inject memory if exists
        if len(plays) > 0:
            surrogate._history.extend(coplays, plays)
            client_player._history.extend(plays, coplays)
        
        # evaluate next move based on given history
        next_action = client_player.strategy(opponent=surrogate)
        plays.append(next_action)
        # assign random action to coplays
        coplays.append(random.choice([Action.C, Action.D]))
        
    print("Finished test run")
    plays_list = "| ".join(play.name for play in plays)
    print("Actions of player one:" + plays_list)
    coplays_list = "| ".join(coplay.name for coplay in coplays)
    print("Actions of player two:" + coplays_list)
    print("----------------")
    

def main():

    # initialize player
    client_player = axl.StochasticWSLS(0)
    #client_player.name = "Random"
    #client_player = axl.GTFT(p=0.0)
    #client_player.name = "T4T"
    client_player.set_seed(42)
    
    res_aware_client_player = ResourceAwareMemOnePlayer(player_instance=copy.deepcopy(client_player),
                                                        resource_scaling_func=util.linear_scaling,
                                                        initial_resource_value=util.ResourceLevel.FULL.value)
    
    test_run(res_aware_client_player)
    
    
if __name__ == '__main__':
    main()