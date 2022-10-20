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
 - [x] Add file line mode
 - [x] Add Line number mode 
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
$ flopth -m alexnet -i 3 224 224 # support model names in torchvision.models
Op AdaptiveAvgPool2d is not supported at now.
+---------------+-------------------+-------------+-------------+----------+----------+-----------------+---------------------------------+
| module_name   | module_type       | in_shape    | out_shape   |   params | flops    | flops_percent   | flops_percent_vis               |
+===============+===================+=============+=============+==========+==========+=================+=================================+
| features.0    | Conv2d            | (3,224,224) | (64,55,55)  |    23296 | 70.4704M | 9.84851%        | #########                       |
+---------------+-------------------+-------------+-------------+----------+----------+-----------------+---------------------------------+
| features.1    | ReLU              | (64,55,55)  | (64,55,55)  |        0 | 193.6K   | 0.0270564%      |                                 |
+---------------+-------------------+-------------+-------------+----------+----------+-----------------+---------------------------------+
| features.2    | MaxPool2d         | (64,55,55)  | (64,27,27)  |        0 | 193.6K   | 0.0270564%      |                                 |
+---------------+-------------------+-------------+-------------+----------+----------+-----------------+---------------------------------+
| features.3    | Conv2d            | (64,27,27)  | (192,27,27) |   307392 | 224.089M | 31.3173%        | ############################### |
+---------------+-------------------+-------------+-------------+----------+----------+-----------------+---------------------------------+
| features.4    | ReLU              | (192,27,27) | (192,27,27) |        0 | 139.968K | 0.0195611%      |                                 |
+---------------+-------------------+-------------+-------------+----------+----------+-----------------+---------------------------------+
| features.5    | MaxPool2d         | (192,27,27) | (192,13,13) |        0 | 139.968K | 0.0195611%      |                                 |
+---------------+-------------------+-------------+-------------+----------+----------+-----------------+---------------------------------+
| features.6    | Conv2d            | (192,13,13) | (384,13,13) |   663936 | 112.205M | 15.6811%        | ###############                 |
+---------------+-------------------+-------------+-------------+----------+----------+-----------------+---------------------------------+
| features.7    | ReLU              | (384,13,13) | (384,13,13) |        0 | 64.896K  | 0.00906947%     |                                 |
+---------------+-------------------+-------------+-------------+----------+----------+-----------------+---------------------------------+
| features.8    | Conv2d            | (384,13,13) | (256,13,13) |   884992 | 149.564M | 20.9021%        | ####################            |
+---------------+-------------------+-------------+-------------+----------+----------+-----------------+---------------------------------+
| features.9    | ReLU              | (256,13,13) | (256,13,13) |        0 | 43.264K  | 0.00604631%     |                                 |
+---------------+-------------------+-------------+-------------+----------+----------+-----------------+---------------------------------+
| features.10   | Conv2d            | (256,13,13) | (256,13,13) |   590080 | 99.7235M | 13.9368%        | #############                   |
+---------------+-------------------+-------------+-------------+----------+----------+-----------------+---------------------------------+
| features.11   | ReLU              | (256,13,13) | (256,13,13) |        0 | 43.264K  | 0.00604631%     |                                 |
+---------------+-------------------+-------------+-------------+----------+----------+-----------------+---------------------------------+
| features.12   | MaxPool2d         | (256,13,13) | (256,6,6)   |        0 | 43.264K  | 0.00604631%     |                                 |
+---------------+-------------------+-------------+-------------+----------+----------+-----------------+---------------------------------+
| avgpool       | AdaptiveAvgPool2d | (256,6,6)   | (256,6,6)   |        0 | 0.0      | 0.0%            |                                 |
+---------------+-------------------+-------------+-------------+----------+----------+-----------------+---------------------------------+
| classifier.0  | Dropout           | (9216)      | (9216)      |        0 | 0.0      | 0.0%            |                                 |
+---------------+-------------------+-------------+-------------+----------+----------+-----------------+---------------------------------+
| classifier.1  | Linear            | (9216)      | (4096)      | 37752832 | 37.7487M | 5.27553%        | #####                           |
+---------------+-------------------+-------------+-------------+----------+----------+-----------------+---------------------------------+
| classifier.2  | ReLU              | (4096)      | (4096)      |        0 | 4.096K   | 0.000572432%    |                                 |
+---------------+-------------------+-------------+-------------+----------+----------+-----------------+---------------------------------+
| classifier.3  | Dropout           | (4096)      | (4096)      |        0 | 0.0      | 0.0%            |                                 |
+---------------+-------------------+-------------+-------------+----------+----------+-----------------+---------------------------------+
| classifier.4  | Linear            | (4096)      | (4096)      | 16781312 | 16.7772M | 2.34468%        | ##                              |
+---------------+-------------------+-------------+-------------+----------+----------+-----------------+---------------------------------+
| classifier.5  | ReLU              | (4096)      | (4096)      |        0 | 4.096K   | 0.000572432%    |                                 |
+---------------+-------------------+-------------+-------------+----------+----------+-----------------+---------------------------------+
| classifier.6  | Linear            | (4096)      | (1000)      |  4097000 | 4.096M   | 0.572432%       |                                 |
+---------------+-------------------+-------------+-------------+----------+----------+-----------------+---------------------------------+


FLOPs: 715.543M
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
This program is mostly inspired by [torchstat](https://github.com/Swall0w/torchstat), great thanks to the creators of it.
