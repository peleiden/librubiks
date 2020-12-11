import os
from dataclasses import dataclass
import functools
import json
import numpy as np
import torch

cpu = torch.device("cpu")
gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reset_cuda():
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def no_grad(fun):
    functools.wraps(fun)
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            return fun(*args, **kwargs)
    return wrapper

