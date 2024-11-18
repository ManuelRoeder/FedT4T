# FedT4T: Resource-Aware Cooperation in Federated Learning

This respository is an implementation of the algorithm proposed in the paper "**Resource-Aware Cooperation in Federated Learning**".

![racf](/assets/racfl.png)

This code was built on [Flower: A Friendly Federated Learning Framework](https://github.com/adap/flower) and [Axelrod: A research tool for the Iterated Prisoner's Dilemma](https://github.com/Axelrod-Python/Axelrod).


# Dependencies

We've tested the code on Python 3.10.15, Pytorch 2.5.0 and TorchVision 0.20.0 with the following dependencies:
```
axelrod==4.13.1
flwr==1.12.0
flwr_datasets==0.4.0
matplotlib==3.9.2
networkx==3.2.1
numpy==2.1.3
pandas==2.2.3
Pillow==11.0.0
seaborn==0.13.2
```

# Framework Usage
FedT4T currently supports resource-awareness of all strategies from the Axelrod framework that are derived from axelrod.MemoryOne by simply wrapping around the instantiated decision rule:
```python
import axelrod
from ipd_client import FedT4TClient
from ipd_player import ResourceAwareMemOnePlayer

# initialize the Axelrod strategy
my_memory_one_strategy = axelrod.FirmButFair()
# use our wrapper to inject resource-awareness
my_resource_aware_strategy = ResourceAwareMemOnePlayer(strategy)
...
# finaly pass the strategy to our Flower client
flower_fedt4t_client = FedT4TClient(..., ipd_strategy=my_resource_aware_strategy, ...)
```


# Running the experiments

# Bibliography
If you find our work to be useful in your research, please cite:
```
```