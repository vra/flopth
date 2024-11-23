import sys

sys.path.insert(0, "../flopth")

import torch

from flopth import flopth
import unittest


class Test(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_transformer(self):
        transformer_model = torch.nn.Transformer(nhead=16, num_encoder_layers=12)
        sum_flops = flopth(transformer_model, in_size=[[32, 512], [32, 512]])[0]
        self.assertTrue(sum_flops == "757.76K")

        transformer_model = torch.nn.TransformerDecoderLayer(d_model=512, nhead=8)
        sum_flops = flopth(transformer_model, in_size=[[32, 512], [32, 512]])[0]
        self.assertTrue(sum_flops == "51.2K")

        transformer_model = torch.nn.TransformerEncoderLayer(d_model=512, nhead=8)
        sum_flops = flopth(transformer_model, in_size=[[32, 512]])[0]
        print("s4:", sum_flops)
        self.assertTrue(sum_flops == "34.816K")
