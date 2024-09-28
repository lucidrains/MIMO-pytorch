import torch
from torch import nn
from torch.nn import Module, ModuleList

import einx
from einops import rearrange, repeat, pack, unpack

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# classes

class MIMO(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
