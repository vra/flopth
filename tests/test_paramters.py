import unittest
import sys

sys.path.insert(0, "../flopth")
from flopth import flopth

import torch.nn as nn


class ModelWithParams(nn.Module):
    def __init__(self, i=0, f=0.0, s="", b=True, num=0):
        super(ModelWithParams, self).__init__()
        self.i = i
        self.f = f
        self.s = s
        self.i = b
        self.num = num
        if i == 0 and f == 0.0 and s == "" and b:
            self.conv1 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(3, 3, kernel_size=3, padding=1)

    def forward(self, x):
        if self.num == 0:
            x = self.conv1(x)
            return x
        if self.num == 1:
            x = self.conv2(x)
            return x
        if self.num == 2:
            x1 = self.conv1(x)
            x2 = self.conv2(x)
            return (x1, x2)
        return None


class Test(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_model_with_param_num0(self):
        model = ModelWithParams(0, 0.0, "", True, num=0)
        sum_flops = flopth(model, in_size=(3, 224, 224))[0]
        self.assertEqual(sum_flops, "4.21478M")

    def test_model_with_param_num1(self):
        model = ModelWithParams(0, 0.0, "", True, num=1)
        sum_flops = flopth(model, in_size=(3, 224, 224))[0]
        self.assertEqual(sum_flops, "4.21478M")

    def test_model_with_param_num2(self):
        model = ModelWithParams(0, 0.0, "", True, num=2)
        sum_flops = flopth(model, in_size=(3, 224, 224))[0]
        self.assertEqual(sum_flops, "8.42957M")


if __name__ == "__main__":
    unittest.main()
