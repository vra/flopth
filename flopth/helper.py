""" Useful tools. Stolen from here: https://github.com/Swall0w/torchstat"""
import numpy as np
import torch
import torch.nn as nn


def compute_flops(module, inp, out):
    if isinstance(module, nn.Conv1d):
        return compute_Conv1d_flops(module, inp[0], out)
    if isinstance(module, nn.Conv2d):
        return compute_Conv2d_flops(module, inp[0], out)
    if isinstance(module, nn.Conv3d):
        return compute_Conv3d_flops(module, inp[0], out)
    elif isinstance(module, nn.BatchNorm1d):
        return compute_BatchNorm1d_flops(module, inp[0], out)
    elif isinstance(module, nn.BatchNorm2d):
        return compute_BatchNorm2d_flops(module, inp[0], out)
    elif isinstance(module, nn.BatchNorm3d):
        return compute_BatchNorm3d_flops(module, inp[0], out)
    elif isinstance(module, (nn.AvgPool2d, nn.MaxPool2d)):
        return compute_Pool2d_flops(module, inp[0], out)
    elif isinstance(module, (nn.ReLU, nn.ReLU6, nn.PReLU, nn.ELU, nn.LeakyReLU)):
        return compute_ReLU_flops(module, inp[0], out)
    elif isinstance(module, nn.Upsample):
        return compute_Upsample_flops(module, inp[0], out)
    elif isinstance(module, nn.Linear):
        return compute_Linear_flops(module, inp[0], out)
    else:
        print("Op {} is not supported at now.".format(module.__class__.__name__))
        return 0
    pass


def compute_Conv1d_flops(module, inp, out):
    # Can have multiple inputs, getting the first one
    assert isinstance(module, nn.Conv1d)
    assert len(inp.size()) == 3 and len(inp.size()) == len(out.size())

    batch_size = inp.size()[0]
    in_c = inp.size()[1]
    k_h = module.kernel_size[0]
    out_c, out_h = out.size()[1:]
    groups = module.groups

    filters_per_channel = out_c // groups
    conv_per_position_flops = k_h * in_c * filters_per_channel
    active_elements_count = batch_size * out_h

    total_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0
    if module.bias is not None:
        bias_flops = out_c * active_elements_count

    total_flops = total_conv_flops + bias_flops
    return total_flops


def compute_Conv2d_flops(module, inp, out):
    # Can have multiple inputs, getting the first one
    assert isinstance(module, nn.Conv2d)
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())

    batch_size = inp.size()[0]
    in_c = inp.size()[1]
    k_h, k_w = module.kernel_size
    out_c, out_h, out_w = out.size()[1:]
    groups = module.groups

    filters_per_channel = out_c // groups
    conv_per_position_flops = k_h * k_w * in_c * filters_per_channel
    active_elements_count = batch_size * out_h * out_w

    total_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0
    if module.bias is not None:
        bias_flops = out_c * active_elements_count

    total_flops = total_conv_flops + bias_flops
    return total_flops


def compute_Conv3d_flops(module, inp, out):
    # Can have multiple inputs, getting the first one
    assert isinstance(module, nn.Conv3d)
    assert len(inp.size()) == 5 and len(inp.size()) == len(out.size())

    batch_size = inp.size()[0]
    in_c = inp.size()[1]
    k_h, k_w, k_d = module.kernel_size
    out_c, out_h, out_w, out_d = out.size()[1:]
    groups = module.groups

    filters_per_channel = out_c // groups
    conv_per_position_flops = k_h * k_w * k_d * in_c * filters_per_channel
    active_elements_count = batch_size * out_h * out_w * out_d

    total_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0
    if module.bias is not None:
        bias_flops = out_c * active_elements_count

    total_flops = total_conv_flops + bias_flops
    return total_flops


def compute_BatchNorm1d_flops(module, inp, out):
    assert isinstance(module, nn.BatchNorm1d)
    assert len(inp.size()) == 3 and len(inp.size()) == len(out.size())
    batch_flops = np.prod(inp.shape)
    if module.affine:
        batch_flops *= 2
    return batch_flops


def compute_BatchNorm2d_flops(module, inp, out):
    assert isinstance(module, nn.BatchNorm2d)
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())
    batch_flops = np.prod(inp.shape)
    if module.affine:
        batch_flops *= 2
    return batch_flops


def compute_BatchNorm3d_flops(module, inp, out):
    assert isinstance(module, nn.BatchNorm3d)
    assert len(inp.size()) == 5 and len(inp.size()) == len(out.size())
    batch_flops = np.prod(inp.shape)
    if module.affine:
        batch_flops *= 2
    return batch_flops


def compute_ReLU_flops(module, inp, out):
    assert isinstance(module, (nn.ReLU, nn.ReLU6, nn.PReLU, nn.ELU, nn.LeakyReLU))
    batch_size = inp.size()[0]
    active_elements_count = batch_size

    for s in inp.size()[1:]:
        active_elements_count *= s

    return active_elements_count


def compute_Pool2d_flops(module, inp, out):
    assert isinstance(module, nn.MaxPool2d) or isinstance(module, nn.AvgPool2d)
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())
    return np.prod(inp.shape)


def compute_Linear_flops(module, inp, out):
    assert isinstance(module, nn.Linear)
    assert len(inp.size()) == 2 and len(out.size()) == 2
    batch_size = inp.size()[0]
    return batch_size * inp.size()[1] * out.size()[1]


def compute_Upsample_flops(module, inp, out):
    assert isinstance(module, nn.Upsample)
    output_size = out[0]
    batch_size = inp.size()[0]
    output_elements_count = batch_size
    for s in output_size.shape[1:]:
        output_elements_count *= s

    return output_elements_count
