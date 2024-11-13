"""
MIT License

Copyright (c) 2024 Manuel Roeder

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import util

# Axelrod framework imports
from axelrod.strategies import MemoryOnePlayer
from axelrod.player import Player
from axelrod.action import Action


class ClientShadowPlayer(Player):
    name = "FL Client Shadow Player"


class RandomMemOnePlayer(MemoryOnePlayer):
    #name = "Random Memory One Player"
    def __init__(
    ) -> None:
        init_action = Action.C#MemoryOnePlayer._random.random_choice(0.5)
        super().__init__((0.5, 0.5, 0.5, 0.5), init_action)
        
        

class ResourceAwareMemOnePlayer:
    '''
        Modification wrapper to make Player types resource-awareS.
        Can be applied to all sochastic memory-one strategies of the Axelrod python library. 
    '''
    def __init__(self, player_instance, resource_scaling_func = util.linear_scaling, initial_resource_level=util.ResourceLevel.FULL.value):
        if not isinstance(player_instance, MemoryOnePlayer):
            raise TypeError("player_instance must be an instance of Player or its subclass")
        player_instance.name = "Res.M1 | " + player_instance.name # resource aware memory one player + axl strategy name
        self.__dict__['_player'] = player_instance
        self._res_lvl = initial_resource_level
        self._res_scaling_func = resource_scaling_func
        
    def __getattr__(self, name):
        """
        Delegate attribute access to the wrapped player instance
        if it's not defined in the wrapper itself.
        """
        return getattr(self._player, name)
    
     # Implement __getstate__ to return the entire state for pickling
    def __getstate__(self):
        return self.__dict__

    # Implement __setstate__ to restore the entire state during unpickling
    def __setstate__(self, state):
        self.__dict__.update(state)
    
    def strategy(self, opponent: Player) -> Action:
        """
        Overwrite MemOne strategy and integrate resource-awareness
        """
        if len(opponent.history) == 0:
            return self._initial
        # Determine which probability to use
        p = self._four_vector[(self.history[-1], opponent.history[-1])]
        # recalculate scaling factor
        scaling_factor = self._res_scaling_func(self._res_lvl)
        p = scaling_factor * p
        # Draw a random number in [0, 1] to decide
        try:
            return self._random.random_choice(p)
        except AttributeError:
            return Action.D if p == 0 else Action.C
        
    def get_resource_level(self):
        return self._res_lvl
    
    def set_resource_level(self, res_lvl):
        self._res_lvl = res_lvl
        
    def get_scaling_func(self):
        return self._res_scaling_func
        
    def set_scaling_func(self, new_scaling_func):
        self._res_scaling_func = new_scaling_func

    
    
    