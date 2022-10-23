<pre align=center style='color:green'>

    ______            __  __  
   / __/ /___  ____  / /_/ /_ 
  / /_/ / __ \/ __ \/ __/ __ \
 / __/ / /_/ / /_/ / /_/ / / /
/_/ /_/\____/ .___/\__/_/ /_/ 
           /_/                

</pre>

# flopth

A simple program to calculate and visualize the FLOPs and Parameters of Pytorch models, with cli tool and Python API.

# Features
 - Handy cli command to show flops and params quickly
 - Visualization percent of flops and params in each layer
 - Support multiple inputs in model's `forward` function
 - Support Both CPU and GPU mode
 - Support Torchscript Model (Only Parameters are shown)
 - Support Python3.5 and above

# Installation
Install stable version of flopth via pypi:
```bash
pip install flopth 
```

or install latest version via github:
```bash
pip install -U git+https://github.com/vra/flopth.git
```

# Usage examples
## cli command
flopth provide cli command `flopth` after installation. You can use it to get information of pytorch models quickly
### Running on models in torchvision.models
with `flopth -m <model_name>`, flopth gives you all information about the `<model_name>`, input shape, output shape, parameter and flops of each layer, and total flops and params.

Here is an example running on alexnet (default input size in (3, 224, 224)):
```plain
$ flopth -m alexnet 
+---------------+-------------------+-------------+-------------+----------+------------------+--------------------------------+----------+-----------------+---------------------+
| module_name   | module_type       | in_shape    | out_shape   | params   | params_percent   | params_percent_vis             | flops    | flops_percent   | flops_percent_vis   |
+===============+===================+=============+=============+==========+==================+================================+==========+=================+=====================+
| features.0    | Conv2d            | (3,224,224) | (64,55,55)  | 23.296K  | 0.0381271%       |                                | 70.4704M | 9.84839%        | ####                |
+---------------+-------------------+-------------+-------------+----------+------------------+--------------------------------+----------+-----------------+---------------------+
| features.1    | ReLU              | (64,55,55)  | (64,55,55)  | 0.0      | 0.0%             |                                | 193.6K   | 0.027056%       |                     |
+---------------+-------------------+-------------+-------------+----------+------------------+--------------------------------+----------+-----------------+---------------------+
| features.2    | MaxPool2d         | (64,55,55)  | (64,27,27)  | 0.0      | 0.0%             |                                | 193.6K   | 0.027056%       |                     |
+---------------+-------------------+-------------+-------------+----------+------------------+--------------------------------+----------+-----------------+---------------------+
| features.3    | Conv2d            | (64,27,27)  | (192,27,27) | 307.392K | 0.50309%         |                                | 224.089M | 31.3169%        | ###############     |
+---------------+-------------------+-------------+-------------+----------+------------------+--------------------------------+----------+-----------------+---------------------+
| features.4    | ReLU              | (192,27,27) | (192,27,27) | 0.0      | 0.0%             |                                | 139.968K | 0.0195608%      |                     |
+---------------+-------------------+-------------+-------------+----------+------------------+--------------------------------+----------+-----------------+---------------------+
| features.5    | MaxPool2d         | (192,27,27) | (192,13,13) | 0.0      | 0.0%             |                                | 139.968K | 0.0195608%      |                     |
+---------------+-------------------+-------------+-------------+----------+------------------+--------------------------------+----------+-----------------+---------------------+
| features.6    | Conv2d            | (192,13,13) | (384,13,13) | 663.936K | 1.08662%         |                                | 112.205M | 15.6809%        | #######             |
+---------------+-------------------+-------------+-------------+----------+------------------+--------------------------------+----------+-----------------+---------------------+
| features.7    | ReLU              | (384,13,13) | (384,13,13) | 0.0      | 0.0%             |                                | 64.896K  | 0.00906935%     |                     |
+---------------+-------------------+-------------+-------------+----------+------------------+--------------------------------+----------+-----------------+---------------------+
| features.8    | Conv2d            | (384,13,13) | (256,13,13) | 884.992K | 1.44841%         |                                | 149.564M | 20.9018%        | ##########          |
+---------------+-------------------+-------------+-------------+----------+------------------+--------------------------------+----------+-----------------+---------------------+
| features.9    | ReLU              | (256,13,13) | (256,13,13) | 0.0      | 0.0%             |                                | 43.264K  | 0.00604624%     |                     |
+---------------+-------------------+-------------+-------------+----------+------------------+--------------------------------+----------+-----------------+---------------------+
| features.10   | Conv2d            | (256,13,13) | (256,13,13) | 590.08K  | 0.965748%        |                                | 99.7235M | 13.9366%        | ######              |
+---------------+-------------------+-------------+-------------+----------+------------------+--------------------------------+----------+-----------------+---------------------+
| features.11   | ReLU              | (256,13,13) | (256,13,13) | 0.0      | 0.0%             |                                | 43.264K  | 0.00604624%     |                     |
+---------------+-------------------+-------------+-------------+----------+------------------+--------------------------------+----------+-----------------+---------------------+
| features.12   | MaxPool2d         | (256,13,13) | (256,6,6)   | 0.0      | 0.0%             |                                | 43.264K  | 0.00604624%     |                     |
+---------------+-------------------+-------------+-------------+----------+------------------+--------------------------------+----------+-----------------+---------------------+
| avgpool       | AdaptiveAvgPool2d | (256,6,6)   | (256,6,6)   | 0.0      | 0.0%             |                                | 9.216K   | 0.00128796%     |                     |
+---------------+-------------------+-------------+-------------+----------+------------------+--------------------------------+----------+-----------------+---------------------+
| classifier.0  | Dropout           | (9216)      | (9216)      | 0.0      | 0.0%             |                                | 0.0      | 0.0%            |                     |
+---------------+-------------------+-------------+-------------+----------+------------------+--------------------------------+----------+-----------------+---------------------+
| classifier.1  | Linear            | (9216)      | (4096)      | 37.7528M | 61.7877%         | ############################## | 37.7487M | 5.27547%        | ##                  |
+---------------+-------------------+-------------+-------------+----------+------------------+--------------------------------+----------+-----------------+---------------------+
| classifier.2  | ReLU              | (4096)      | (4096)      | 0.0      | 0.0%             |                                | 4.096K   | 0.000572425%    |                     |
+---------------+-------------------+-------------+-------------+----------+------------------+--------------------------------+----------+-----------------+---------------------+
| classifier.3  | Dropout           | (4096)      | (4096)      | 0.0      | 0.0%             |                                | 0.0      | 0.0%            |                     |
+---------------+-------------------+-------------+-------------+----------+------------------+--------------------------------+----------+-----------------+---------------------+
| classifier.4  | Linear            | (4096)      | (4096)      | 16.7813M | 27.4649%         | #############                  | 16.7772M | 2.34465%        | #                   |
+---------------+-------------------+-------------+-------------+----------+------------------+--------------------------------+----------+-----------------+---------------------+
| classifier.5  | ReLU              | (4096)      | (4096)      | 0.0      | 0.0%             |                                | 4.096K   | 0.000572425%    |                     |
+---------------+-------------------+-------------+-------------+----------+------------------+--------------------------------+----------+-----------------+---------------------+
| classifier.6  | Linear            | (4096)      | (1000)      | 4.097M   | 6.70531%         | ###                            | 4.096M   | 0.572425%       |                     |
+---------------+-------------------+-------------+-------------+----------+------------------+--------------------------------+----------+-----------------+---------------------+


FLOPs: 715.553M
Params: 61.1008M
```

### Running on custom models
Also, given model name and the file path where the model defined, flopth will output model information:

For the dummpy network `MyModel` defined in `/tmp/my_model.py`,
```python
# file path: /tmp/my_model.py
# model name:  MyModel
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(3, 3, kernel_size=3, padding=1)

    def forward(self, x1):
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x1 = self.conv4(x1)
        return x1
```
You can use `flopth -m MyModel -p /tmp/my_model -i 3 224 224` to print model information:

```plain
$ flopth -m MyModel -p /tmp/my_model.py -i 3 224 224
+---------------+---------------+-------------+-------------+----------+------------------+----------------------+----------+-----------------+---------------------+
| module_name   | module_type   | in_shape    | out_shape   |   params | params_percent   | params_percent_vis   | flops    | flops_percent   | flops_percent_vis   |
+===============+===============+=============+=============+==========+==================+======================+==========+=================+=====================+
| conv1         | Conv2d        | (3,224,224) | (3,224,224) |       84 | 25.0%            | ############         | 4.21478M | 25.0%           | ############        |
+---------------+---------------+-------------+-------------+----------+------------------+----------------------+----------+-----------------+---------------------+
| conv2         | Conv2d        | (3,224,224) | (3,224,224) |       84 | 25.0%            | ############         | 4.21478M | 25.0%           | ############        |
+---------------+---------------+-------------+-------------+----------+------------------+----------------------+----------+-----------------+---------------------+
| conv3         | Conv2d        | (3,224,224) | (3,224,224) |       84 | 25.0%            | ############         | 4.21478M | 25.0%           | ############        |
+---------------+---------------+-------------+-------------+----------+------------------+----------------------+----------+-----------------+---------------------+
| conv4         | Conv2d        | (3,224,224) | (3,224,224) |       84 | 25.0%            | ############         | 4.21478M | 25.0%           | ############        |
+---------------+---------------+-------------+-------------+----------+------------------+----------------------+----------+-----------------+---------------------+

FLOPs: 16.8591M
Params: 336.0
```

#### Multiple inputs
If your model has more than one input in `forward`, you can add multiple `-i` parameters to flopth:

```python
# file path: /tmp/my_model.py
# model name:  MyModel
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(3, 3, kernel_size=3, padding=1)

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        x2 = self.conv3(x2)
        x2 = self.conv4(x2)
        return (x1, x2)
```
You can use `flopth -m MyModel -p /tmp/my_model -i 3 224 224 -i 3 128 128` to print model information:

```plain
 flopth -m MyModel -p /tmp/my_model.py  -i 3 224 224 -i 3 128 128
+---------------+---------------+-------------+-------------+----------+------------------+----------------------+----------+-----------------+---------------------+
| module_name   | module_type   | in_shape    | out_shape   |   params | params_percent   | params_percent_vis   | flops    | flops_percent   | flops_percent_vis   |
+===============+===============+=============+=============+==========+==================+======================+==========+=================+=====================+
| conv1         | Conv2d        | (3,224,224) | (3,224,224) |       84 | 25.0%            | ############         | 4.21478M | 37.6923%        | ##################  |
+---------------+---------------+-------------+-------------+----------+------------------+----------------------+----------+-----------------+---------------------+
| conv2         | Conv2d        | (3,224,224) | (3,224,224) |       84 | 25.0%            | ############         | 4.21478M | 37.6923%        | ##################  |
+---------------+---------------+-------------+-------------+----------+------------------+----------------------+----------+-----------------+---------------------+
| conv3         | Conv2d        | (3,128,128) | (3,128,128) |       84 | 25.0%            | ############         | 1.37626M | 12.3077%        | ######              |
+---------------+---------------+-------------+-------------+----------+------------------+----------------------+----------+-----------------+---------------------+
| conv4         | Conv2d        | (3,128,128) | (3,128,128) |       84 | 25.0%            | ############         | 1.37626M | 12.3077%        | ######              |
+---------------+---------------+-------------+-------------+----------+------------------+----------------------+----------+-----------------+---------------------+


FLOPs: 11.1821M
Params: 336.0
```

#### Extra arguments in model's initialization
flopth with options like `-x param1=int:3 param2=float:5.2` to process the extra parameters in model's initialization:
```python
# file path: /tmp/my_model.py
# model name:  MyModel
import torch.nn as nn


class MyModel(nn.Module):
    # Please Notice the parameters ks1 and ks2 here!
    def __init__(self, ks1, ks2):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=ks1, padding=1)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=ks1, padding=1)
        self.conv3 = nn.Conv2d(3, 3, kernel_size=ks2, padding=1)
        self.conv4 = nn.Conv2d(3, 3, kernel_size=ks2, padding=1)

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        x2 = self.conv3(x2)
        x2 = self.conv4(x2)
        return (x1, x2)
```
In order to pass value to the arguments of `ks1` and `ks2`, we can run flopth like this:
```plain
$ flopth -m MyModel -p /tmp/my_model.py -i 3 224 224 -i 3 128 128 -x ks1=int:3 ks2=int:1
+---------------+---------------+-------------+-------------+----------+------------------+-----------------------+----------+-----------------+-------------------------+
| module_name   | module_type   | in_shape    | out_shape   |   params | params_percent   | params_percent_vis    | flops    | flops_percent   | flops_percent_vis       |
+===============+===============+=============+=============+==========+==================+=======================+==========+=================+=========================+
| conv1         | Conv2d        | (3,224,224) | (3,224,224) |       84 | 43.75%           | ##################### | 4.21478M | 47.6707%        | ####################### |
+---------------+---------------+-------------+-------------+----------+------------------+-----------------------+----------+-----------------+-------------------------+
| conv2         | Conv2d        | (3,224,224) | (3,224,224) |       84 | 43.75%           | ##################### | 4.21478M | 47.6707%        | ####################### |
+---------------+---------------+-------------+-------------+----------+------------------+-----------------------+----------+-----------------+-------------------------+
| conv3         | Conv2d        | (3,128,128) | (3,130,130) |       12 | 6.25%            | ###                   | 202.8K   | 2.29374%        | #                       |
+---------------+---------------+-------------+-------------+----------+------------------+-----------------------+----------+-----------------+-------------------------+
| conv4         | Conv2d        | (3,130,130) | (3,132,132) |       12 | 6.25%            | ###                   | 209.088K | 2.36486%        | #                       |
+---------------+---------------+-------------+-------------+----------+------------------+-----------------------+----------+-----------------+-------------------------+


FLOPs: 8.84146M
Params: 192.0
```

### Line number mode
One of the fancy features of flopth is that given the line number where the model **object** is definited, flopth can print model information:
```python
# file path: /tmp/my_model.py
# model name:  MyModel
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(3, 3, kernel_size=3, padding=1)

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        x2 = self.conv3(x2)
        x2 = self.conv4(x2)
        return (x1, x2)


if __name__ == '__main__':
    my_model = MyModel()
```

Since the model object `my_model` in defined in line 23, we can run flopth like this:
```plain
$ flopth -n 23 -p /tmp/my_model.py -i 3 224 224 -i 3 128 128
+---------------+---------------+-------------+-------------+----------+------------------+----------------------+----------+-----------------+---------------------+
| module_name   | module_type   | in_shape    | out_shape   |   params | params_percent   | params_percent_vis   | flops    | flops_percent   | flops_percent_vis   |
+===============+===============+=============+=============+==========+==================+======================+==========+=================+=====================+
| conv1         | Conv2d        | (3,224,224) | (3,224,224) |       84 | 25.0%            | ############         | 4.21478M | 37.6923%        | ##################  |
+---------------+---------------+-------------+-------------+----------+------------------+----------------------+----------+-----------------+---------------------+
| conv2         | Conv2d        | (3,224,224) | (3,224,224) |       84 | 25.0%            | ############         | 4.21478M | 37.6923%        | ##################  |
+---------------+---------------+-------------+-------------+----------+------------------+----------------------+----------+-----------------+---------------------+
| conv3         | Conv2d        | (3,128,128) | (3,128,128) |       84 | 25.0%            | ############         | 1.37626M | 12.3077%        | ######              |
+---------------+---------------+-------------+-------------+----------+------------------+----------------------+----------+-----------------+---------------------+
| conv4         | Conv2d        | (3,128,128) | (3,128,128) |       84 | 25.0%            | ############         | 1.37626M | 12.3077%        | ######              |
+---------------+---------------+-------------+-------------+----------+------------------+----------------------+----------+-----------------+---------------------+


FLOPs: 11.1821M
Params: 336.0
```

**Notice: Although line number mode of flopth is quite handy, it may fail when the model definition is too complex, e.g., using outer config file to initialize a model. In this case, I recommend you to use flopth's Python API detailed below.**

## Python API
The Python API of flopth is quite simple:
```python
import torch
import torch.nn as nn

# import flopth
from flopth import flopth

# define Model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(3, 3, kernel_size=3, padding=1)

    def forward(self, x1):
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x1 = self.conv4(x1)
        return x1


# declare Model object
my_model = MyModel()

# Use input size
flops, params = flopth(my_model, in_size=((3, 224, 224),))
print(flops, params)

# Or use input tensors
dummy_inputs = torch.rand(1, 3, 224, 224)
flops, params = flopth(my_model, inputs=(dummy_inputs,))
print(flops, params)
```

The output is like this:
```plain
16.8591M 336.0
```

To show detail information of each layer, add `show_detail=True` in flopth function call:
```python
flops, params = flopth(my_model, in_size=((3, 224, 224),), show_detail=True)
```

The outputs:
```plain
+---------------+---------------+-------------+-------------+----------+------------------+----------------------+----------+-----------------+---------------------+
| module_name   | module_type   | in_shape    | out_shape   |   params | params_percent   | params_percent_vis   | flops    | flops_percent   | flops_percent_vis   |
+===============+===============+=============+=============+==========+==================+======================+==========+=================+=====================+
| conv1         | Conv2d        | (3,224,224) | (3,224,224) |       84 | 25.0%            | ############         | 4.21478M | 25.0%           | ############        |
+---------------+---------------+-------------+-------------+----------+------------------+----------------------+----------+-----------------+---------------------+
| conv2         | Conv2d        | (3,224,224) | (3,224,224) |       84 | 25.0%            | ############         | 4.21478M | 25.0%           | ############        |
+---------------+---------------+-------------+-------------+----------+------------------+----------------------+----------+-----------------+---------------------+
| conv3         | Conv2d        | (3,224,224) | (3,224,224) |       84 | 25.0%            | ############         | 4.21478M | 25.0%           | ############        |
+---------------+---------------+-------------+-------------+----------+------------------+----------------------+----------+-----------------+---------------------+
| conv4         | Conv2d        | (3,224,224) | (3,224,224) |       84 | 25.0%            | ############         | 4.21478M | 25.0%           | ############        |
+---------------+---------------+-------------+-------------+----------+------------------+----------------------+----------+-----------------+---------------------+


16.8591M 336.0
```

To show only the value of flops and params (no unit conversion), add `bare_number=True` to flopth function call:
```python
flops, params = flopth(my_model, in_size=((3, 224, 224),), bare_number=True)
```

The outputs:
```plain
16859136 336
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

# TODOs
 - [x] Support multiple inputs
 - [x] Add parameter size
 - [x] Add file line mode
 - [x] Add line number mode 
 - [ ] Support more modules 

# Contribution and issue
Any discussion and contribution are very welcomed. Please open an issue to reach me. 

# Acknowledge
This program is mostly inspired by [torchstat](https://github.com/Swall0w/torchstat), great thanks to the creators of it.
