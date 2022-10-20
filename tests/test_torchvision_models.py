import sys

sys.path.insert(0, "../flopth")
from flopth import flopth
import unittest

import torchvision.models as models


class Test(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_alexnet(self):
        alexnet = models.alexnet()
        sum_flops = flopth(alexnet, in_size=(3, 224, 224))[0]
        self.assertTrue(sum_flops == "715.553M")

    def test_densenet121(self):
        model = models.densenet121()
        sum_flops = flopth(model, in_size=(3, 224, 224))[0]
        self.assertTrue(sum_flops == "2.88262G")

    def test_densenet161(self):
        model = models.densenet161()
        sum_flops = flopth(model, in_size=(3, 224, 224))[0]
        self.assertTrue(sum_flops == "7.81836G")

    def test_densenet169(self):
        model = models.densenet169()
        sum_flops = flopth(model, in_size=(3, 224, 224))[0]
        self.assertTrue(sum_flops == "3.41836G")

    def test_densenet201(self):
        model = models.densenet201()
        sum_flops = flopth(model, in_size=(3, 224, 224))[0]
        self.assertTrue(sum_flops == "4.36697G")

    def test_inception(self):
        model = models.inception_v3()
        sum_flops = flopth(model, in_size=(3, 224, 224))[0]
        self.assertTrue(sum_flops == "2.84855G")

    def test_resnet18(self):
        model = models.resnet18()
        sum_flops = flopth(model, in_size=(3, 224, 224))[0]
        self.assertTrue(sum_flops == "1.82142G")

    def test_resnet34(self):
        model = models.resnet34()
        sum_flops = flopth(model, in_size=(3, 224, 224))[0]
        self.assertTrue(sum_flops == "3.67425G")

    def test_resnet50(self):
        model = models.resnet50()
        sum_flops = flopth(model, in_size=(3, 224, 224))[0]
        self.assertTrue(sum_flops == "4.11864G")

    def test_resnet101(self):
        model = models.resnet101()
        sum_flops = flopth(model, in_size=(3, 224, 224))[0]
        self.assertTrue(sum_flops == "7.84451G")

    def test_resnet152(self):
        model = models.resnet152()
        sum_flops = flopth(model, in_size=(3, 224, 224))[0]
        self.assertTrue(sum_flops == "11.5736G")

    def test_squeeze_1_0(self):
        model = models.squeezenet1_0()
        sum_flops = flopth(model, in_size=(3, 224, 224))[0]
        self.assertTrue(sum_flops == "830.046M")

    def test_squeeze_1_1(self):
        model = models.squeezenet1_1()
        sum_flops = flopth(model, in_size=(3, 224, 224))[0]
        self.assertTrue(sum_flops == "355.862M")

    def test_vgg11(self):
        model = models.vgg11()
        sum_flops = flopth(model, in_size=(3, 224, 224))[0]
        self.assertTrue(sum_flops == "7.6301G")

    def test_vgg13(self):
        model = models.vgg13()
        sum_flops = flopth(model, in_size=(3, 224, 224))[0]
        self.assertTrue(sum_flops == "11.3391G")

    def test_vgg16(self):
        model = models.vgg16()
        sum_flops = flopth(model, in_size=(3, 224, 224))[0]
        self.assertTrue(sum_flops == "15.5035G")

    def test_vgg19(self):
        model = models.vgg19()
        sum_flops = flopth(model, in_size=(3, 224, 224))[0]
        self.assertTrue(sum_flops == "19.6679G")


if __name__ == "__main__":
    unittest.main()
