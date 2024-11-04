from enum import Enum
from axelrod.action import Action

class ResourceLevel(float, Enum):
    NONE = -1.0
    LOW = 0.0
    MODERATE = 0.5
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