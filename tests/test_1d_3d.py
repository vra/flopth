import unittest
import sys

sys.path.insert(0, "../flopth")
from flopth import flopth

import torch.nn as nn
import torchvision.models as models


class Model1D(nn.Module):
    def __init__(self):
        super(Model1D, self).__init__()
        self.conv1 = nn.Conv1d(3, 10, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv1(x)


class Model3D(nn.Module):
    def __init__(self):
        super(Model3D, self).__init__()
        self.conv1 = nn.Conv3d(3, 10, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv1(x)


class Test(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_model_1d(self):
        model = Model1D()
        sum_flops = flopth(model, in_size=(3, 224))[0]
        self.assertEqual(sum_flops, "22.4K")

    def test_model_3d(self):
        model = Model3D()
        sum_flops = flopth(model, in_size=(3, 1, 224, 224))[0]
        self.assertEqual(sum_flops, "41.1443M")


if __name__ == "__main__":
    unittest.main()
