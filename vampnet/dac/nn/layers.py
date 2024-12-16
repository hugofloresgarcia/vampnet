import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))

import cached_conv as cc
def CausalConv1d(*args, **kwargs):
    # grab the padding 
    kernel_size=kwargs.get('kernel_size', 1)
    stride=kwargs.get('stride', 1)
    dilation=kwargs.get('dilation', 1)
    kwargs['padding'] = cc.get_padding(
        kernel_size=kernel_size, 
        stride=stride, 
        dilation=dilation, 
        mode="causal"
    )

    return cc.Conv1d(*args, **kwargs)

def CausalConvTranspose1d(*args, **kwargs):
    return cc.ConvTranspose1d(*args, **kwargs, causal=True)


# Scripting this brings model speed up 1.4x
@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)
