import numpy as np


def divide_by_unit(value):
    if isinstance(value, np.ndarray):
        value = float(value)
    if value > 1e9:
        return "{:.6}G".format(value / 1e9)
    elif value > 1e6:
        return "{:.6}M".format(value / 1e6)
    elif value > 1e3:
        return "{:.6}K".format(value / 1e3)
    return "{:.6}".format(value / 1.0)
