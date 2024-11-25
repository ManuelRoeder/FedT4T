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

import random
from enum import Enum
from axelrod.action import Action
import numpy as np

# global seed
SEED = 42
random.seed(SEED)

class ResourceLevel(float, Enum):
    NONE = -1.0
    EMPTY = 0.0 + 0.1E-6
    LOW = 0.25
    MODERATE = 0.5
    HIGH = 0.75
    FULL = 1.0
    
    @classmethod
    def from_float(cls, value: float):
        """
        Maps a float to the corresponding ResourceLevel enum member.
        
        Parameters:
        - value (float): The float value to map to a ResourceLevel.
        
        Returns:
        - ResourceLevel: The matching enum member, or raises ValueError if not found.
        """
        try:
            return cls(value)
        except ValueError:
            raise ValueError(f"No matching ResourceLevel for value: {value}")
    
    def to_string(self):
            """
            Converts the ResourceLevel enum member to its string name.
            
            Returns:
            - str: The name of the ResourceLevel member.
            """
            return self.name
        
    
def generate_hash(x_id, y_id, use_cantor=False):
    sorted_x, sorted_y = sorted([x_id, y_id])
    if use_cantor:
        return sorted_x, sorted_y, str(cantor_pairing(sorted_x, sorted_y))
    else:
        # simple concat str
        hash_str = str(sorted_x) + "_" + str(sorted_y)
        return sorted_x, sorted_y, hash_str

def decode_hash(hash_str, use_cantor=False):
    if use_cantor:
        x, y = reverse_cantor_pairing(hash_str)
        return str(x), str(y)
    else:
        str_lst = hash_str.split("_")
        return str_lst[0], str_lst[1]

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


def actions_to_string(actions):
    # Convert each Action to its name (or value, depending on your preference)
    return ''.join(action.name for action in actions)
        
    
def string_to_actions(action_str):
    # Map each character in the string to the corresponding Action enum
    action_map = {
        'D': Action.D,
        'C': Action.C
    }
    return [action_map[char] for char in action_str]



# linear scaling function for resource level
def linear_scaling(res_lvl):
    # normalize and scale
    e_scaled = (res_lvl - ResourceLevel.EMPTY.value) / (ResourceLevel.FULL.value - ResourceLevel.EMPTY.value)
    return e_scaled


def synergy_threshold_scaling(res_lvl, gamma=8):
    """
    Computes the value of the function:
    f_res(res_lvl) = 0.5 * (1 + tanh(gamma * (res_lvl - E_low)))
    
    Parameters:
    - res_lvl: Input energy or value (array or scalar)
    - E_low: Lower energy threshold
    - gamma: Scaling factor controlling the steepness around E_low

    Returns:
    - The computed result of the function
    """
    return 0.5 * (1 + np.tanh(gamma * (res_lvl - ResourceLevel.LOW.value)))
    
    


def random_action_choice(p: float = 0.5) -> Action:
        """
        Return C with probability `p`, else return D

        No random sample is carried out if p is 0 or 1.

        Parameters
        ----------
        p : float
            The probability of picking C

        Returns
        -------
        axelrod.Action
        """
        if p == 0:
            return Action.D

        if p == 1:
            return Action.C

        r = random.uniform(0.0, 1.0)
        if r < p:
            return Action.C
        return Action.D
