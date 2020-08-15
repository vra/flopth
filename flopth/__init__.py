import argparse
import importlib
import os
import sys

import numpy as np
import torch
import torchvision
import torchvision.models as models

from flopth.model_viewer import ModelViewer
from flopth.settings import settings


def parse_parameters():
    parser = argparse.ArgumentParser('A program to calculate FLOPs for pytorch models')
    parser.add_argument('-p', '--module_path', default=None,
                        help="Path to the .py file which contains model" +
                        "definition, e.g., ../lib/my_models.py")
    parser.add_argument('class_name',
                        help="Name of the net to evaluate defined in " +
                        "modeule_path, e.g., DeepLab")
    parser.add_argument('-d', '--dtype', default="float32",
                        help="Type of input tensor of target net. default is " +
                        "float32")
    parser.add_argument('-i', '--in_size', nargs="+", type=int, action='append',
                        help="Input size of target net, without batch_size " +
                        "multiple inputs supported, e.g., -i 3 224 224 -i 3 112 112")

    parser.add_argument('--show_detail', default=True, action="store_true")
    parser.add_argument('--bare_number', default=False, action="store_true")
    return parser.parse_args()


def parse_net(module_path, class_name):
    torchvision_models = [m for m in torchvision.models.__dict__.keys() if '__' not in m]
    if class_name in torchvision_models:
        Model = getattr(torchvision.models, class_name)
    else:
        fail_msg = 'For the model not in torchvision.models, you have to ' + \
            'specify the python file where the model is defined.'
        assert module_path is not None, fail_msg
        module_dir = os.path.dirname(module_path)
        module_name = os.path.basename(module_path).split('.')[0]
        sys.path.insert(0, module_dir)
        custom_models = importlib.import_module(module_name)
        Model = getattr(custom_models, class_name)

    model = Model()

    return model


def main():
    args = parse_parameters()

    model = parse_net(args.module_path, args.class_name)

    sum_flops = flopth(model, in_size=args.in_size, dtype=args.dtype, param_dict=settings.param_dict, show_detail=args.show_detail, bare_number=args.bare_number)
    param_size = sum(np.prod(v.size()) for v in model.parameters()) / 1e6
    out_info = 'FLOPs: {}\nParam size: {:.5}M'.format(sum_flops, param_size)
    print(out_info)


def flopth(model, in_size, dtype='float32', param_dict=settings.param_dict, show_detail=False, bare_number=False):
    dtype = getattr(torch, dtype)
    input_list = []
    for size in in_size:
        x = torch.rand([1, *size], dtype=dtype)
        input_list.append(x)

    mv = ModelViewer(model, input_list, param_dict, dtype)

    if show_detail:
        mv.show_info(show_detail=True)
    if bare_number:
        return mv.get_info(for_human=False)
    else:
        return mv.get_info()


if __name__ == '__main__':
    main()
