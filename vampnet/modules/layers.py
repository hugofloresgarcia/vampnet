import time
from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils import weight_norm


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


def num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def recurse_children(module, fn):
    for child in module.children():
        if isinstance(child, nn.ModuleList):
            for c in child:
                yield recurse_children(c, fn)
        if isinstance(child, nn.ModuleDict):
            for c in child.values():
                yield recurse_children(c, fn)

        yield recurse_children(child, fn)
        yield fn(child)


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class CodebookEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        latent_dim: int,
        n_codebooks: int,
        emb_dim: int,
        special_tokens: Optional[Tuple[str]] = None,
        memory_tokens: int = 0,
    ):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.emb_dim = emb_dim
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size


        if special_tokens is None:
            special_tokens = []

        # add memory tokens to special tokens
        if memory_tokens > 0:
            special_tokens += [f"<MEMORY_{i}>" for i in range(memory_tokens)]
        

        self.special = nn.ParameterDict(
            {
                tkn: nn.Parameter(torch.randn(n_codebooks, self.latent_dim))
                for tkn in special_tokens
            }
        )
        self.special_idxs = {
            tkn: i + vocab_size for i, tkn in enumerate(special_tokens)
        }

        self.out_proj = nn.Conv1d(n_codebooks * self.latent_dim, self.emb_dim, 1)

    def from_codes(self, codes: torch.Tensor, codec=None):
        """
        get a sequence of continuous embeddings from a sequence of discrete codes.
        unlike it's counterpart in the original VQ-VAE, this function adds for any special tokens
        necessary for the language model, like <MASK>.
        """
        n_codebooks = codes.shape[1]
        latent = []
        for i in range(n_codebooks):
            c = codes[:, i, :]

            if torch.any(codes < self.vocab_size):
                assert codec is not None, f"Codec must be provided for codec tokens"
                lookup_table = codec.quantizer.quantizers[i].codebook.weight
                if hasattr(self, "special"):
                    special_lookup = torch.cat(
                        [self.special[tkn][i : i + 1] for tkn in self.special], dim=0
                    )
                    lookup_table = torch.cat([lookup_table, special_lookup], dim=0)

                l = F.embedding(c, lookup_table).transpose(1, 2)
                latent.append(l)
            else:
                c = c - self.vocab_size
                assert hasattr(self, "special"), f"Special tokens must be provided"
                special_lookup = torch.cat(
                    [self.special[tkn][i : i + 1] for tkn in self.special], dim=0
                )
                l = F.embedding(c, special_lookup).transpose(1, 2)
                latent.append(l) 

        latent = torch.cat(latent, dim=1)
        return latent

    def forward(self, latents: torch.Tensor):
        """
        project a sequence of latents to a sequence of embeddings
        """
        x = self.out_proj(latents)
        return x
