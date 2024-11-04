from collections import OrderedDict
from typing import Dict, Tuple
from typing import Optional
from logging import INFO
from flwr.common.logger import log
import torch
from flwr.common import NDArrays, Scalar, Config
from flwr.client import NumPyClient
from model import Net
import axelrod as axl
from axelrod.action import Action
from ipd_player import ClientShadowPlayer, ResourceAwareMemOnePlayer
import util

class FlowerClient(NumPyClient):
    def __init__(self, trainloader, valloader, ipd_strategy: axl.Player, client_id) -> None:
        super().__init__()

        self.trainloader = trainloader
        self.valloader = valloader
        self.model = Net(num_classes=10)
        self.ipd_strategy = ipd_strategy
        self.client_id = client_id

    def fit(self, parameters, config):
        """This method trains the model using the parameters sent by the
        server on the dataset of this client. At then end, the parameters
        of the locally trained model are communicated back to the server"""

        # copy parameters sent by the server into client's local model
        set_params(self.model, parameters)
        
        # enforce cooperate/defect decision here
        match_id, cooperate = self.evaluate_pd(config)
        
        # prepare return meta data
        # check client resource level
        res_level = util.ResourceLevel.NONE
        if isinstance(self.ipd_strategy, ResourceAwareMemOnePlayer):
            res_level = self.ipd_strategy.get_resource_level()
            
        ret_dict = {"match_id": match_id, "client_id": self.client_id, "ipd_strategy_name": self.ipd_strategy.name, "resource_level": res_level.value}
        
        if cooperate:
            log(INFO, "Client Id %s fit(): COOPERATE action", self.client_id)
            # Define the optimizer
            optim = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)

            # do local training (call same function as centralised setting)
            train(self.model, self.trainloader, optim, epochs=1)
            
            # return the model parameters to the server as well as extra info (number of training examples in this case)
            return get_params(self.model), len(self.trainloader), ret_dict
        else:
            log(INFO, "Client Id %s fit(): DEFECT action", self.client_id)
            
        # return empty answer to signal defect
        return get_params(self.model), 0, ret_dict

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        """Evaluate the model sent by the server on this client's
        local validation set. Then return performance metrics."""

        set_params(self.model, parameters)
        # do local evaluation (call same function as centralised setting)
        loss, accuracy = test(self.model, self.valloader)
        # send statistics back to the server
        return float(loss), len(self.valloader), {"accuracy": accuracy}
    
    def evaluate_pd(self, config: Dict[str, Scalar]):
        """Function that evaluates to cooperate or defect based on the internal
           IPD strategy and results from previous matchups."""
        ipd_history_list_plays = list()
        ipd_history_list_coplays = list()
        match_id = 0
        # get the match results from the configuration as integer
        log(INFO, "Checking properties")
        if "ipd_history_plays" in config.keys():
            ipd_history_str_plays = config["ipd_history_plays"]
            log(INFO, "Play history found: %s", ipd_history_str_plays)
            ipd_history_list_plays = util.string_to_actions(ipd_history_str_plays)
        if "ipd_history_coplays" in config.keys():
            ipd_history_str_coplays = config["ipd_history_coplays"]
            log(INFO, "Co-Play history found: %s", ipd_history_str_coplays)
            ipd_history_list_coplays = util.string_to_actions(ipd_history_str_coplays)
        if "match_id" in config.keys():
            match_id = config["match_id"]
            log(INFO, "Match id found %s", match_id)
            
        # always reset strategy history
        self.ipd_strategy.reset()
        
        # sanity check on matchup history
        if len(ipd_history_list_plays) > 0:
            if len(ipd_history_list_coplays) == len(ipd_history_list_plays):
                print("Sanity check on matchup history success")
                # restore internal history
                self.ipd_strategy._history.extend(ipd_history_list_plays, ipd_history_list_coplays)
            else:
                print("Sanity check on matchup history failure")
        else:
            print("No matchup history found, sanity check success")
            
        # create shadow opponent
        shadow_opponent = ClientShadowPlayer()
        
        if len(ipd_history_list_plays) > 0:
            # assign flipped plays / coplays
            shadow_opponent._history.extend(ipd_history_list_coplays, ipd_history_list_plays)
        # evaluate next move based on given history
        next_action = self.ipd_strategy.strategy(opponent=shadow_opponent)
        
        # convert to coop / defect
        action = True if next_action == axl.Action.C else False
        
        # return act
        return match_id, action
    
    def get_properties(self, config: Config) -> dict[str, Scalar]:
        """Return a client's set of properties.

        Parameters
        ----------
        config : Config
            Configuration parameters requested by the server.
            This can be used to tell the client which properties
            are needed along with some Scalar attributes.

        Returns
        -------
        properties : Dict[str, Scalar]
            A dictionary mapping arbitrary string keys to values of type
            bool, bytes, float, int, or str. It can be used to communicate
            arbitrary property values back to the server.
        """
        retDict = dict()
        if "client_id" in config.keys():
            retDict["client_id"] = self.client_id
        if "strategy" in config.keys():
            retDict["strategy"] = self.ipd_strategy.name
        return retDict
            

def bool_list_to_action_list(bool_list):
    out_list = list()
    for b in bool_list:
        if b:
            out_list.append(axl.Action.C)
        else:
            out_list.append(axl.Action.D)

def int_to_bool_list(n):
    # Convert the integer to its binary representation, remove the '0b' prefix
    binary_rep = bin(n)[2:]
    
    # Convert each digit to a boolean value
    bool_list = [bool(int(digit)) for digit in binary_rep]
    
    return bool_list

def int_to_action_list(n):
    # Convert the integer to its binary representation, remove the '0b' prefix
    binary_rep = bin(n)[2:]
    
    # Convert each digit to action list
    action_list = [(Action.C if bool(int(digit)) else Action.D) for digit in binary_rep]
    
    return action_list

def append_bool_to_msb(n, new_bool):
    # Find the number of bits in the integer
    num_bits = n.bit_length()
    
    # Shift the integer left by 1 to make space for the new MSB
    n = n << 1
    
    # If the new boolean is True, set the most significant bit to 1
    if new_bool:
        n += 1 << num_bits  # Add 1 at the MSB position
    
    return n


# Two auxhiliary functions to set and extract parameters of a model
def set_params(model, parameters):
    """Replace model parameters with those passed as `parameters`."""

    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    # now replace the parameters
    model.load_state_dict(state_dict, strict=True)


def get_params(model):
    """Extract model parameters as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def train(net, trainloader, optimizer, epochs):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    for batch in trainloader:
        images, labels = batch["image"], batch["label"]
        optimizer.zero_grad()
        loss = criterion(net(images), labels)
        loss.backward()
        optimizer.step()
        

def train_iter(net, data_iterator, optimizer, epochs):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    batch = next(data_iterator)
    images, labels = batch["image"], batch["label"]
    optimizer.zero_grad()
    loss = criterion(net(images), labels)
    loss.backward()
    optimizer.step()


def test(net, testloader):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["image"], batch["label"]
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy