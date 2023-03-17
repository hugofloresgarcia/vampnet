import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo
    (identical to OpenAI GPT). Also see the Gaussian Error Linear Units
    paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
                )
            )
        )

class GatedGELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gelu = NewGELU()

    def forward(self, x, dim: int = -1):
        p1, p2 = x.chunk(2, dim=dim)
        return p1 * self.gelu(p2)

class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(channels))

    def forward(self, x):
        return x + (self.alpha + 1e-9).reciprocal() * torch.sin(self.alpha * x).pow(2)

def get_activation(name: str = "relu"):
    if name == "relu":
        return nn.ReLU
    elif name == "gelu":
        return NewGELU
    elif name == "geglu":
        return GatedGELU
    elif name == "snake":
        return Snake1d
    else:
        raise ValueError(f"Unrecognized activation {name}")