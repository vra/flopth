import unittest
import sys

sys.path.insert(0, "../flopth")
from flopth import flopth

import torch.nn as nn


class DictOutputModel(nn.Module):
    def __init__(self):
        super(DictOutputModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        return {"x1": x1, "x2": x2}


class TupleOutputModel(nn.Module):
    def __init__(self):
        super(TupleOutputModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        return (x1, x2)


class MyOutput:
    def __init__(self, x1, x2):
        self.x1 = x1
        self.x2 = x2


class CustomClassOutputModel(nn.Module):
    def __init__(self):
        super(CustomClassOutputModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        return MyOutput(x1, x2)


class Test(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_dict_output_model(self):
        model = DictOutputModel()
        sum_flops = flopth(model, in_size=(3, 224, 224))[0]
        self.assertEqual(sum_flops, "8.42957M")

    def test_tuple_output_model(self):
        model = TupleOutputModel()
        sum_flops = flopth(model, in_size=(3, 224, 224))[0]
        self.assertEqual(sum_flops, "8.42957M")

    def test_custom_class_output_model(self):
        model = CustomClassOutputModel()
        sum_flops = flopth(model, in_size=(3, 224, 224))[0]
        self.assertEqual(sum_flops, "8.42957M")


if __name__ == "__main__":
    unittest.main()
