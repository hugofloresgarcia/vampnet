import time
from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils import weight_norm


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


class SequentialWithFiLM(nn.Module):
    """
    handy wrapper for nn.Sequential that allows FiLM layers to be
    inserted in between other layers.
    """

    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    @staticmethod
    def has_film(module):
        mod_has_film = any(
            [res for res in recurse_children(module, lambda c: isinstance(c, FiLM))]
        )
        return mod_has_film

    def forward(self, x, cond):
        for layer in self.layers:
            if self.has_film(layer):
                x = layer(x, cond)
            else:
                x = layer(x)
        return x


class FiLM(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        if input_dim > 0:
            self.beta = nn.Linear(input_dim, output_dim)
            self.gamma = nn.Linear(input_dim, output_dim)

    def forward(self, x, r):
        if self.input_dim == 0:
            return x
        else:
            beta, gamma = self.beta(r), self.gamma(r)
            beta, gamma = (
                beta.view(x.size(0), self.output_dim, 1),
                gamma.view(x.size(0), self.output_dim, 1),
            )
            x = x * (gamma + 1) + beta
        return x


class CodebookEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        latent_dim: int,
        n_codebooks: int,
        emb_dim: int,
        special_tokens: Optional[Tuple[str]] = None,
    ):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.emb_dim = emb_dim
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size

        if special_tokens is not None:
            for tkn in special_tokens:
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

    def from_codes(self, codes: torch.Tensor, codec):
        n_codebooks = codes.shape[1]
        latent = []
        for i in range(n_codebooks):
            c = codes[:, i, :]

            lookup_table = codec.quantizer.quantizers[i].codebook.weight
            if hasattr(self, "special"):
                special_lookup = torch.cat(
                    [self.special[tkn][i : i + 1] for tkn in self.special], dim=0
                )
                lookup_table = torch.cat([lookup_table, special_lookup], dim=0)

            l = F.embedding(c, lookup_table).transpose(1, 2)
            latent.append(l)

        latent = torch.cat(latent, dim=1)
        return latent

    def forward(self, latents: torch.Tensor):
        x = self.out_proj(latents)
        return x

    def flatten(self, tokens: torch.Tensor, n_codebooks: int = None):
        n_c = n_codebooks if n_codebooks is not None else self.n_codebooks
        return rearrange(tokens, "b c t -> b (t c)", c=n_c)

    def unflatten(self, flat_tokens: torch.Tensor, n_codebooks: int = None):
        nb, nt = flat_tokens.shape

        n_c = n_codebooks if n_codebooks is not None else self.n_codebooks
        tokens = rearrange(flat_tokens, "b (t c) -> b c t", c=n_c)

        return tokens
