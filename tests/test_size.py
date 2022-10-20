import unittest
import sys
import torch.nn as nn

sys.path.insert(0, "../flopth")
from flopth import flopth  # noqa


class ModelKernel1(nn.Module):
    def __init__(self):
        super(ModelKernel1, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=1, padding=1)

    def forward(self, x):
        return self.conv1(x)


class ModelKernel3(nn.Module):
    def __init__(self):
        super(ModelKernel3, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv1(x)


class ModelKernel5(nn.Module):
    def __init__(self):
        super(ModelKernel5, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=5, padding=1)

    def forward(self, x):
        return self.conv1(x)


class ModelKernel7(nn.Module):
    def __init__(self):
        super(ModelKernel7, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=7, padding=1)

    def forward(self, x):
        return self.conv1(x)


class Test(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_model_kernel_1_112(self):
        model = ModelKernel1()
        sum_flops = flopth(model, in_size=(3, 112, 112))[0]
        self.assertEqual(sum_flops, "155.952K")

    def test_model_kernel_1(self):
        model = ModelKernel1()
        sum_flops = flopth(model, in_size=(3, 224, 224))[0]
        self.assertEqual(sum_flops, "612.912K")

    def test_model_kernel_3(self):
        model = ModelKernel3()
        sum_flops = flopth(model, in_size=(3, 224, 224))[0]
        self.assertEqual(sum_flops, "4.21478M")

    def test_model_kernel_5(self):
        model = ModelKernel5()
        sum_flops = flopth(model, in_size=(3, 224, 224))[0]
        self.assertEqual(sum_flops, "11.2368M")

    def test_model_kernel_7(self):
        model = ModelKernel7()
        sum_flops = flopth(model, in_size=(3, 224, 224))[0]
        self.assertEqual(sum_flops, "21.4896M")


if __name__ == "__main__":
    unittest.main()
