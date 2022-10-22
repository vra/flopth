import unittest
import sys

sys.path.insert(0, "../flopth")
from flopth import flopth

import torch
import torch.nn as nn
import torchvision.models as models


class OneInputOneOutputModel(nn.Module):
    def __init__(self):
        super(OneInputOneOutputModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv1(x)


class MultipleInputOneOutputModel(nn.Module):
    def __init__(self):
        super(MultipleInputOneOutputModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, padding=1)

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x2 = self.conv1(x2)
        x = x1 + x2
        return x


class OneInputMultipleOutputModel(nn.Module):
    def __init__(self):
        super(OneInputMultipleOutputModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        return [x1, x2]


class MultipleInputMultipleOutputModel(nn.Module):
    def __init__(self):
        super(MultipleInputMultipleOutputModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(3, 3, kernel_size=3, padding=1)

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        x2 = self.conv3(x2)
        x2 = self.conv4(x2)
        return [x1, x2]


class Test(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_one_input_one_output(self):
        model = OneInputOneOutputModel()
        sum_flops = flopth(model, in_size=(3, 224, 224))[0]
        self.assertEqual(sum_flops, "4.21478M")

    def test_one_input_one_output_tuple_input(self):
        model = OneInputOneOutputModel()
        sum_flops = flopth(model, in_size=((3, 224, 224),))[0]
        self.assertEqual(sum_flops, "4.21478M")

    def test_one_input_one_output_default_in_size(self):
        model = OneInputOneOutputModel()
        sum_flops = flopth(model)[0]
        self.assertEqual(sum_flops, "4.21478M")

    def test_multiple_input_one_output(self):
        model = MultipleInputOneOutputModel()
        sum_flops = flopth(model, in_size=[(3, 224, 224), (3, 224, 224)])[0]
        self.assertEqual(sum_flops, "4.21478M")

    def test_one_input_mulitple_output(self):
        model = OneInputMultipleOutputModel()
        sum_flops = flopth(model, in_size=(3, 224, 224))[0]
        self.assertEqual(sum_flops, "8.42957M")

    def test_multiple_input_mulitple_output(self):
        model = MultipleInputMultipleOutputModel()
        sum_flops = flopth(model, in_size=[(3, 224, 224), (3, 224, 224)])[0]
        self.assertEqual(sum_flops, "16.8591M")

    def test_multiple_input_mulitple_output_list_of_list_input(self):
        model = MultipleInputMultipleOutputModel()
        sum_flops = flopth(model, in_size=[[3, 224, 224], [3, 224, 224]])[0]
        self.assertEqual(sum_flops, "16.8591M")

    def test_multiple_input_mulitple_output_tuple_of_tuple_input(self):
        model = MultipleInputMultipleOutputModel()
        sum_flops = flopth(model, in_size=((3, 224, 224), (3, 224, 224)))[0]
        self.assertEqual(sum_flops, "16.8591M")

    def test_tensor_variable_inputs(self):
        model = MultipleInputMultipleOutputModel()
        dummpy_input1 = torch.rand(1, 3, 224, 224)
        dummpy_input2 = torch.rand(1, 3, 224, 224)
        sum_flops = flopth(model, inputs=(dummpy_input1, dummpy_input2))[0]
        self.assertEqual(sum_flops, "16.8591M")


if __name__ == "__main__":
    unittest.main()
