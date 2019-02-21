# flopth
A simple program to calculate the FLOPs of Pytorch models, with cli tool and API.

flopth can run in CPU mode and GPU mode. multi-gpu is also supported. Also, flopth support extra parameters in model's `forward` function. Besides, flopth support python3.5, while some other tools only support python3.6+.

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
```bash
# Example1
flopth alexnet  # support model names in torchvision.models

# Example2
# -p for the path to the python file where MySOATModel defined, -i for input size, -x for extra parameters
flopth MySOATModel -p /path/to/the/python/file/where/class/MySOATModel/is/defined/models.py -i 3 224 224 -x 1994025
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

sum_flops = flopth.flopth(m, in_size=[10], extra_params=233)
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
 Will give wrong FLOPs value, because of we use [register_buffer ](https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module.register_buffer), which is bind to a `nn.Module` (in this example, `l1`). 

# Acknowledge
This program is mostly inspired by [torchstat](https://github.com/Swall0w/torchstat), great thanks to the creator of it.
