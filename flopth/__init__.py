import argparse
import importlib
import os
from pydoc import locate
import sys

import numpy as np
import torch
import torchvision
import torchvision.models as models

from flopth.model_viewer import ModelViewer
from flopth.settings import settings
from flopth.utils import divide_by_unit


def parse_parameters():
    parser = argparse.ArgumentParser(
        "A program to calculate FLOPs and #Parameters of pytorch models\n\n"
    )
    parser.add_argument(
        "-p",
        "--module_path",
        default=None,
        help="Path to a .py file which contains pytorch model definition, e.g., ../lib/my_models.py",
    )
    parser.add_argument(
        "-n",
        "--line_number",
        default=None,
        type=int,
        help="Line number contains model" + " definition, e.g., 10",
    )
    parser.add_argument(
        "-m",
        "--class_name",
        default=None,
        type=str,
        help="Name of the net to evaluate defined in module_path, e.g., DeepLab",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        default="float32",
        help="Type of input tensor of target net. default is float32",
    )
    parser.add_argument(
        "-i",
        "--in_size",
        nargs="+",
        type=int,
        default=None,
        action="append",
        help="Input size of target net, without batch_size. "
        + "multiple inputs supported, e.g., -i 3 224 224 -i 3 112 112",
    )

    parser.add_argument(
        "-x",
        "--extra_args",
        metavar="KEY=VALUE",
        nargs="+",
        help="extra arguments in model's forward function",
    )

    parser.add_argument(
        "--show_detail",
        default=True,
        action="store_true",
        help="whether to show the detailed flops and params of each layer",
    )

    parser.add_argument(
        "--bare_number",
        default=False,
        action="store_true",
        help="Show raw number, useful when used in python code",
    )

    args = parser.parse_args()
    if args.line_number is None and args.class_name is None:
        parser.print_help()
        sys.exit(0)
    return args


def parse_var(s):
    """
    Parse a key, value pair, separated by '='
    That's the reverse of ShellArgs.

    On the command line (argparse) a declaration will typically look like:
        foo=hello
    or
        foo="hello world"
    """
    items = s.split("=")
    key = items[0].strip()  # we remove blanks around keys, as is logical
    if len(items) > 1:
        # rejoin the rest:
        data_type = items[1].split(":")[0]
        real_type = locate(data_type)
        data_value = items[1].split(":")[1]
        value = real_type(data_value)
    return (key, value)


def parse_vars(items):
    """
    Parse a series of key-value pairs and return a dictionary
    """
    d = {}

    if items:
        for item in items:
            key, value = parse_var(item)
            d[key] = value
    return d


def parse_net(module_path, class_name, line_number, extra_args):
    fail_msg_model_definition = "You must use ONE of line_number and class_name"
    assert not (class_name is None and line_number is None), fail_msg_model_definition

    torchvision_models = [
        m for m in torchvision.models.__dict__.keys() if "__" not in m
    ]

    if class_name in torchvision_models:
        Model = getattr(torchvision.models, class_name)
        model = Model(**extra_args)
    else:
        fail_msg = (
            "For the model not in torchvision.models, you have to "
            + "specify the python file where the model is defined."
        )
        assert module_path is not None, fail_msg
        module_dir = os.path.dirname(module_path)
        sys.path.insert(0, module_dir)
        module_basename = os.path.basename(module_path)
        module_name = module_basename.split(".")[0]
        custom_models = importlib.import_module(module_name)
        if class_name is not None:
            Model = getattr(custom_models, class_name)
            model = Model(**extra_args)
        elif line_number is not None:
            # import all functions in module, need improve in future
            import_line = "from {} import *".format(module_name)
            original_line = open(module_path).readlines()[line_number - 1].strip()
            fail_msg_equal_in_line = (
                "Model definition line must have the format of "
                + '"foo = MyModel(param1=xxx, param2=yyy, ...)", i.e., "=" is needed'
            )
            assert "=" in original_line, original_line

            # convert foo = MyModel(param1, param2,...) to model = MyModel(param1, param2...)
            exec_line = (
                import_line + ";" + "model =" + "=".join(original_line.split("=")[1:])
            )

            loc = {}
            exec(exec_line, globals(), loc)
            model = loc["model"]
            exec(exec_line)

    return model


def main():
    args = parse_parameters()

    extra_args = parse_vars(args.extra_args)
    model = parse_net(args.module_path, args.class_name, args.line_number, extra_args)

    sum_flops, sum_params = flopth(
        model,
        in_size=args.in_size,
        dtype=args.dtype,
        param_dict=settings.param_dict,
        show_detail=args.show_detail,
        bare_number=args.bare_number,
    )

    out_info = "FLOPs: {}\nParams: {}".format(sum_flops, sum_params)
    print(out_info)


def flopth(
    model,
    in_size=[[3, 224, 224]],
    inputs=None,
    dtype="float32",
    param_dict=settings.param_dict,
    show_detail=False,
    bare_number=False,
):
    if in_size is None:
        in_size = [[3, 224, 224]]

    dtype = getattr(torch, dtype)
    if inputs is not None:
        input_list = list(inputs)
    else:
        input_list = []
        if isinstance(in_size, tuple):
            if isinstance(in_size[0], int):
                in_size = [in_size]
            elif isinstance(in_size[0], tuple):
                in_size = list(in_size)
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


if __name__ == "__main__":
    main()
