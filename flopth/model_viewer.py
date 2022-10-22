""" Main file to calculate information of a pytorch model. """
import numpy as np
from tabulate import tabulate
import torch

from flopth.helper import compute_flops
from flopth.utils import divide_by_unit


class ModelViewer:
    def __init__(self, model, input_list, param_dict, dtype="float32"):
        self._model = model

        self.leaf_modules, self.leaf_module_names = self.obtain_leaf_modules()

        # run register and calculate info
        self.register_parameter(param_dict)
        self.apply_forward_hook()
        if torch.cuda.is_available():
            for i in range(len(input_list)):
                input_list[i] = input_list[i].cuda()

            self._model = self._model.cuda()
        self._model.eval()
        self._model(*input_list)

    #        self.show_results()

    def obtain_leaf_modules(self):
        """Get modules which have no children."""
        leaf_modules = []
        leaf_module_names = []
        for n, m in self._model.named_modules():
            if len(list(m.children())) == 0:
                if not isinstance(m, torch.nn.Module):
                    print(
                        "Module {} is not supported at now. All info about it will be ignored.".format(
                            n
                        )
                    )
                    continue
                leaf_modules.append(m)
                leaf_module_names.append(n)

        return leaf_modules, leaf_module_names

    def register_parameter(self, param_dict):
        """Register Parameters to leaf nn.Module instance in the model.

        Args:
            param_dict: see `param_dict` in `settings.py` for details.
        """
        assert "flops" in param_dict.keys(), 'The key "flops" must be in params'
        for m in self.leaf_modules:
            for k in param_dict.keys():
                m.register_buffer(
                    k,
                    getattr(
                        torch.zeros(param_dict[k]["size"]), param_dict[k]["type"]
                    )(),
                )

    def apply_forward_hook(self):
        def forward_with_hook(module, *args, **kwargs):
            # Calculate FLOPs of current module
            output = self.forward_funcs[module.__class__](module, *args, **kwargs)
            args_list = list(args)
            module.flops = torch.from_numpy(
                np.array(compute_flops(module, args_list, output), dtype=np.int64)
            )
            return output

        self.forward_funcs = {}
        for m in self.leaf_modules:
            if m.__class__ not in self.forward_funcs.keys():
                self.forward_funcs[m.__class__] = m.__class__.__call__
                m.__class__.__call__ = forward_with_hook

    def show_info(self, show_detail=True):
        sum_flops = torch.zeros((1), dtype=torch.int64)
        if torch.cuda.is_available():
            sum_flops = sum_flops.cuda()
            in_shape = sum_flops[1:6]
            out_shape = sum_flops[6:]
            sum_flops = sum_flops[0]

        for m, n in zip(self.leaf_modules, self.leaf_module_names):
            if torch.cuda.is_available():
                sum_flops += m.flops.cuda()[0].item()
            else:
                sum_flops += m.flops[0].item()
        sum_flops = (
            sum_flops.detach().cpu().numpy().item()
            if sum_flops.is_cuda
            else sum_flops.detach().numpy().item()
        )
        sum_params = sum(np.prod(v.size()) for v in self._model.parameters())

        if show_detail:
            info = []
            for m, n in zip(self.leaf_modules, self.leaf_module_names):
                param = sum(np.prod(v.size()) for v in m.parameters())
                m_type = (
                    str(type(m))
                    .split(">")[0]
                    .split("class ")[1]
                    .strip("'")
                    .split(".")[-1]
                )
                flops = (
                    m.flops.detach().cpu().numpy()
                    if m.flops.is_cuda
                    else m.flops.detach().numpy()
                )

                #                if flops.ndim > 0:
                #                    flops = flops[0]

                in_shape = flops[1:6]
                out_shape = flops[6:]
                flops = flops[0]

                in_shape_str = (
                    "(" + ",".join(str(e) for e in in_shape).split("-1")[0][:-1] + ")"
                )
                out_shape_str = (
                    "(" + ",".join(str(e) for e in out_shape).split("-1")[0][:-1] + ")"
                )

                info.append(
                    [
                        n,
                        m_type,
                        in_shape_str,
                        out_shape_str,
                        divide_by_unit(param),
                        "{:.6}%".format(param / sum_params * 100)
                        if sum_params > 0
                        else "",
                        "#" * int(param / sum_params * 50) if sum_params > 0 else "",
                        divide_by_unit(flops),
                        "{:.6}%".format(flops / sum_flops * 100)
                        if sum_flops > 0
                        else "",
                        "#" * int(flops / sum_flops * 50) if sum_flops > 0 else "",
                    ]
                )
            print(
                tabulate(
                    info,
                    headers=(
                        "module_name",
                        "module_type",
                        "in_shape",
                        "out_shape",
                        "params",
                        "params_percent",
                        "params_percent_vis",
                        "flops",
                        "flops_percent",
                        "flops_percent_vis",
                    ),
                    tablefmt="grid",
                )
            )
            print("\n")

            # sum_flops_str = self.divide_by_unit(sum_flops)
            # print('\nTotal FLOPs: {}'.format(sum_flops_str))

        return sum_flops, sum_params

    def get_info(self, for_human=True):
        flops, params = self.show_info(show_detail=False)
        if for_human:
            flops = divide_by_unit(flops)
            params = divide_by_unit(params)
        return flops, params
