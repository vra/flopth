# flopth
A simple program to calculate the FLOPs of Pytorch models, with cli tool and API.

flopth can run in CPU mode and GPU mode. multi-gpu is also supported. Also, flopth supports multiple inputs in model's `forward` function. Besides, flopth support python3.5, while some other tools only support python3.6+.  

# Features
 - Support Python3.5+
 - Both CPU and GPU mode are supported
 - Multi-GPU mode is supported
 - Support multiple inputs in module's `forward` function

# TODOs
 - [x] Support multiple inputs
 - [x] Add parameter size
 - [ ] Support more modules 

# Installation
```bash
pip install --user flopth 
```
or 
```bash
sudo pip install flopth
```

# Use examples
## cli tool
### Example1
```bash
$ flopth alexnet -i 3 224 224 # support model names in torchvision.models
module        flops           percent       percent-vis
------------  --------------  ------------  -------------------------------
features.0    70.4704 MFlops  9.84851%      #########
features.1    193.6 KFlops    0.0270564%
features.2    193.6 KFlops    0.0270564%
features.3    224.089 MFlops  31.3173%      ###############################
features.4    139.968 KFlops  0.0195611%
features.5    139.968 KFlops  0.0195611%
features.6    112.205 MFlops  15.6811%      ###############
features.7    64.896 KFlops   0.00906947%
features.8    149.564 MFlops  20.9021%      ####################
features.9    43.264 KFlops   0.00604631%
features.10   99.7235 MFlops  13.9368%      #############
features.11   43.264 KFlops   0.00604631%
features.12   43.264 KFlops   0.00604631%
classifier.0  0.0 Flops       0.0%
classifier.1  37.7487 MFlops  5.27553%      #####
classifier.2  4.096 KFlops    0.000572432%
classifier.3  0.0 Flops       0.0%
classifier.4  16.7772 MFlops  2.34468%      ##
classifier.5  4.096 KFlops    0.000572432%
classifier.6  4.096 MFlops    0.572432%


FLOPs: 715.543 MFlops
Param size: 61.101M
```
### Example2
```bash
# -p for the path to the python file where MySOTAModel defined, -i for input size, you can use -i multiple times for multiple inputs
$ flopth MySOTAModel -p /path/to/the/python/file/where/class/MySOTAModel/is/defined/models.py -i 3 224 224 -i 1 224 224
...
```

## Python API
### Example1
```python
from flopth import flopth
import torchvision.models as models

alexnet = models.alexnet()
sum_flops = flopth(alexnet)
print(sum_flops)
```
### Example2
```python
from flopth import flopth
import torch.nn as nn


class TwoLinear(nn.Module):
    def __init__(self):
        super(TwoLinear, self).__init__()

        self.l1 = nn.Linear(10, 1994)
        self.l2 = nn.Linear(1994, 10)

    def forward(self, x, y):
        x = self.l1(x) * y
        x = self.l2(x) * y

        return x


m = TwoLinear()

sum_flops = flopth(m, in_size=[[10], [10]])
print(sum_flops)
```

# Known issues
 1. When use a module more than one time during `forward`, the FLOPs calculation is not correct, For example:
 ```python
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.l1 = nn.Linear(10, 10)

    def forward(self, x, y):
        x = self.l1(x)
        x = self.l1(x)
        x = self.l1(x)

        return x
 ```
 Will give wrong FLOPs value, because we use [register_buffer ](https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module.register_buffer), which is bind to a `nn.Module` (in this example, `l1`). 

# Acknowledge
This program is mostly inspired by [torchstat](https://github.com/Swall0w/torchstat), great thanks to the creator of it.
