import math
import logging
from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import lightning as L
from einops import rearrange

from x_transformers import ContinuousTransformerWrapper
from x_transformers import Encoder

from .activations import get_activation
from .layers import CodebookEmbedding
from .layers import FiLM
from .layers import SequentialWithFiLM
from .layers import WNConv1d
from ..util import scalar_to_batch_tensor, codebook_flatten, codebook_unflatten
from ..mask import _gamma
from .lora import Linear

LORA_R = 8

@torch.jit.script_if_tracing
def gumbel_noise_like(t: Tensor):
    noise = torch.zeros_like(t).uniform_(1e-20, 1.0)
    return -torch.log(-torch.log(noise))


def gumbel_sample(t, temperature=1.0, dim=-1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise_like(t)).argmax(dim=dim)


class VampNet(L.LightningModule):
    def __init__(
        self,
        n_heads: int = 20,
        n_layers: int = 20,
        r_cond_dim: int = 0,
        n_codebooks: int = 4,
        n_conditioning_codebooks: int = 0,
        latent_dim: int = 8,
        embedding_dim: int = 1000,
        vocab_size: int = 1024,
        flash_attn: bool = False,
        noise_mode: str = "mask",
        dropout: float = 0.0
    ):
        super().__init__()
        assert r_cond_dim == 0, f"r_cond_dim must be 0 (not supported), but got {r_cond_dim}"
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.r_cond_dim = r_cond_dim
        self.n_codebooks = n_codebooks
        self.n_conditioning_codebooks = n_conditioning_codebooks
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.flash_attn = flash_attn
        self.noise_mode = noise_mode

        assert self.noise_mode == "mask", "deprecated"

        # add an embedding layer per codebook
        assert embedding_dim % n_codebooks == 0, f"embedding_dim must be divisible by n_codebooks, but got {embedding_dim} and {n_codebooks}"
        self.embedding = nn.Embedding(
            ((vocab_size) * n_codebooks) + 1, embedding_dim // n_codebooks
        )
        self.mask_token = (vocab_size * n_codebooks)

        self.lm = ContinuousTransformerWrapper(
            max_seq_len=2048,
            attn_layers=Encoder(
                dim=self.embedding_dim,
                depth=self.n_layers,
                heads=self.n_heads,
                attn_flash=True,
                ff_glu=True, 
                use_rmsnorm=True, 
                rotary_pos_emb=True
            ),
            use_abs_pos_emb=False,
            emb_dropout=dropout,
        )


        # Add final conv layer
        self.n_predict_codebooks = n_codebooks - n_conditioning_codebooks
        self.classifier = SequentialWithFiLM(
            WNConv1d(
                embedding_dim,
                vocab_size * self.n_predict_codebooks,
                kernel_size=1,
                padding="same",
                # groups=self.n_predict_codebooks,
            ),
        )

        # self.rearrange_bpct_bptc = Rearrange("b (p c) t -> b p (t c)", c=self.n_predict_codebooks)
    
    def codebook_idx_to_global_idx(self, codes: Tensor):
        # codes is shape (b, n_codebooks, t)
        # print a slice of the codes
        mask = codes == self.mask_token
        old_codes = codes.clone()

        code_offsets = torch.arange(self.n_codebooks).to(codes.device) * self.vocab_size
        code_offsets = code_offsets[None, :, None].repeat(codes.shape[0], 1, codes.shape[2])
        codes = codes + code_offsets

        # place the mask token back
        codes[mask] = self.mask_token
        return codes

    def forward(self, x, return_activations: bool = False):
        # input should be shape (batch, codebook, seq)
        x = self.codebook_idx_to_global_idx(x)
        # pass through the embedding layer, output shape (batch, codebook, seq, emb)
        x = self.embedding(x)
        # concat the embds along the codebook dimension

        x = rearrange(x, "b c n d -> b n (c d)")
        x_mask = torch.ones_like(x, dtype=torch.bool)[:, :, :1].squeeze(2)

        assert return_activations == False, "return_activations not supported sry :( would happily implement if you need it"
        out = self.lm(
            x, return_mems=False, 
            mask=x_mask, 
        )

        out = rearrange(out, "b n d -> b d n")

        out = self.classifier(out, None) # no cond here!

        b, pc, t = out.shape
        out = rearrange(out, "b (p c) t -> b p (t c)", c=self.n_predict_codebooks)

        if return_activations:
            return out, activations
        else:
            return out
    
    def r_embed(self, r, max_positions=10000):
        if self.r_cond_dim > 0:
            dtype = r.dtype

            r = _gamma(r) * max_positions
            half_dim = self.r_cond_dim // 2

            emb = math.log(max_positions) / (half_dim - 1)
            emb = torch.arange(half_dim, device=r.device).float().mul(-emb).exp()

            emb = r[:, None] * emb[None, :]
            emb = torch.cat([emb.sin(), emb.cos()], dim=1)

            if self.r_cond_dim % 2 == 1:  # zero pad
                emb = nn.functional.pad(emb, (0, 1), mode="constant")

            return emb.to(dtype)
        else:
            return r

    @torch.inference_mode()
    def generate(
        self, 
        codes: Optional[torch.Tensor] = None,
        sampling_steps: int = 12,
        temperature: float = 1.0,
        mask_temperature: float = 10.5,
        typical_filtering=True,
        typical_mass=0.15,
        typical_min_tokens=64,
        top_p=None,
        seed: int = None,
        sample_cutoff: float = 1.0,
        causal_weight: float = 0.0,
        cfg_guidance: float = None,
    ):
        if seed is not None:
            torch.manual_seed(seed)
        logging.debug(f"beginning generation with {sampling_steps} steps")

        ##################### 
        # resolve initial z #
        #####################
        z = codes
        nb = z.shape[0]

        #################
        # resolve mask #
        #################

        # get the mask from the start tokens
        mask = z == self.mask_token
        print(f"found {mask.sum()} mask tokens in start tokens (total {mask.numel()})")

        ###########
        # set up #
        ##########
        # apply the mask to z
        z_masked = z.masked_fill(mask.bool(), self.mask_token)
        # logging.debug(f"z_masked: {z_masked}")

        # how many mask tokens to begin with?
        num_mask_tokens_at_start = (z_masked == self.mask_token).sum()

        # how many codebooks are we inferring vs conditioning on?
        n_infer_codebooks = self.n_codebooks - self.n_conditioning_codebooks

        if cfg_guidance is not None:
            # we need to repeat our tensors
            z_uncond = torch.full_like(z, self.mask_token)

            z_masked = torch.cat(
                (z_masked, z_uncond), dim=0
            )
            z = torch.cat(
                (z, z_uncond), dim=0
            )
            mask = torch.cat(
                (mask, torch.full_like(mask, 1)), dim=0
            )

        #################
        # begin sampling #
        #################
        from tqdm import tqdm
        for i in range(sampling_steps):

            # our current schedule step
            r = scalar_to_batch_tensor(
                (i + 1) / sampling_steps, 
                z.shape[0]
            ).to(z.device)

            # infer from latents
            # NOTE: this collapses the codebook dimension into the sequence dimension
            logits = self.forward(z_masked) # b, prob, seq

            if cfg_guidance is not None:
                logits_cond, logits_uncond = logits[:nb], logits[nb:]
                logits_cond = cfg_guidance * logits_cond + cfg_guidance * (1 - logits_uncond)

            logits = logits.permute(0, 2, 1)  # b, seq, prob
            b = logits.shape[0]

            sampled_z, selected_probs = sample_from_logits(
                logits, sample=(
                   (i / sampling_steps) <= sample_cutoff
                ), 
                temperature=temperature,
                typical_filtering=typical_filtering, typical_mass=typical_mass,
                typical_min_tokens=typical_min_tokens,
                top_k=None, top_p=top_p, return_probs=True,
            )


            # flatten z_masked and mask, so we can deal with the sampling logic
            # we'll unflatten them at the end of the loop for the next forward pass
            # remove conditioning codebooks, we'll add them back at the end
            z_masked = codebook_flatten(z_masked[:, self.n_conditioning_codebooks:, :])           

            mask = (z_masked == self.mask_token).int()
            
            # update the mask, remove conditioning codebooks from the mask
            # add z back into sampled z where the mask was false
            sampled_z = torch.where(
                mask.bool(), sampled_z, z_masked
            )

            # ignore any tokens that weren't masked
            selected_probs = torch.where(
                mask.bool(), selected_probs, torch.inf
            )

            # get the num tokens to mask, according to the schedule
            num_to_mask = torch.floor(_gamma(r) * num_mask_tokens_at_start).unsqueeze(1).long()
            logging.debug(f"num to mask: {num_to_mask}")

            if i != (sampling_steps - 1):
                num_to_mask = torch.maximum(
                    torch.tensor(1),
                    torch.minimum(
                        mask.sum(dim=-1, keepdim=True) - 1,
                        num_to_mask
                    )
                )


            # get our new mask
            mask = mask_by_random_topk(
                num_to_mask, selected_probs, mask_temperature * (1-r)
            )  

            # update the mask
            z_masked = torch.where(
                mask.bool(), self.mask_token, sampled_z
            )

            z_masked = codebook_unflatten(z_masked, n_infer_codebooks)
            mask = codebook_unflatten(mask, n_infer_codebooks)

            # add conditioning codebooks back to z_masked
            z_masked = torch.cat(
                (z[:, :self.n_conditioning_codebooks, :], z_masked), dim=1
            )

        # add conditioning codebooks back to sampled_z
        sampled_z = codebook_unflatten(sampled_z, n_infer_codebooks)
        sampled_z = torch.cat(
            (z[:, :self.n_conditioning_codebooks, :], sampled_z), dim=1
        )

        if cfg_guidance is not None:
            sampled_z = sampled_z[:nb]

        return sampled_z




        
def sample_from_logits(
        logits, 
        sample: bool = True,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
        typical_filtering: bool = False,
        typical_mass: float = 0.2,
        typical_min_tokens: int = 1,
        return_probs: bool = False
    ):
    """Convenience function to sample from a categorial distribution with input as
    unnormalized logits.

    Parameters
    ----------
    logits : Tensor[..., vocab_size]
    config: SamplingConfig
        The set of hyperparameters to be used for sampling
        sample : bool, optional
            Whether to perform multinomial sampling, by default True
        temperature : float, optional
            Scaling parameter when multinomial samping, by default 1.0
        top_k : int, optional
            Restricts sampling to only `top_k` values acc. to probability,
            by default None
        top_p : float, optional
            Restricts sampling to only those values with cumulative
            probability = `top_p`, by default None

    Returns
    -------
    Tensor[...]
        Sampled tokens
    """
    shp = logits.shape[:-1]

    if typical_filtering:
        typical_filter(logits, 
                        typical_mass=typical_mass, 
                        typical_min_tokens=typical_min_tokens
        )

    # Apply top_k sampling
    if top_k is not None:
        v, _ = logits.topk(top_k)
        logits[logits < v[..., [-1]]] = -float("inf")

    # Apply top_p (nucleus) sampling
    if top_p is not None and top_p < 1.0:
        v, sorted_indices = logits.sort(descending=True)
        cumulative_probs = v.softmax(dim=-1).cumsum(dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        # Right shift indices_to_remove to keep 1st token over threshold
        sorted_indices_to_remove = F.pad(sorted_indices_to_remove, (1, 0), value=False)[
            ..., :-1
        ]

        # Compute indices_to_remove in unsorted array
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )

        logits[indices_to_remove] = -float("inf")

    # Perform multinomial sampling after normalizing logits
    probs = (
        F.softmax(logits / temperature, dim=-1)
        if temperature > 0
        else logits.softmax(dim=-1)
    )
    token = (
        probs.view(-1, probs.size(-1)).multinomial(1).squeeze(1).view(*shp)
        if sample
        else logits.argmax(-1)
    )

    if return_probs:
        token_probs = probs.take_along_dim(token.unsqueeze(-1), dim=-1).squeeze(-1)
        return token, token_probs
    else:
        return token
    


def mask_by_random_topk(
        num_to_mask: int, 
        probs: torch.Tensor, 
        temperature: float = 1.0, 
    ):
    """
    Args:
        num_to_mask (int): number of tokens to mask
        probs (torch.Tensor): probabilities for each sampled event, shape (batch, seq)
        temperature (float, optional): temperature. Defaults to 1.0.
    """
    logging.debug(f"masking by random topk")
    logging.debug(f"num to mask: {num_to_mask}")
    logging.debug(f"probs shape: {probs.shape}")
    logging.debug(f"temperature: {temperature}")
    logging.debug("")

    noise = gumbel_noise_like(probs)
    temperature = temperature.unsqueeze(-1)
    confidence = torch.log(probs) + temperature * noise
    logging.debug(f"confidence shape: {confidence.shape}")

    sorted_confidence, sorted_idx = confidence.sort(dim=-1)
    logging.debug(f"sorted confidence shape: {sorted_confidence.shape}")
    logging.debug(f"sorted idx shape: {sorted_idx.shape}")

    # get the cut off threshold, given the mask length
    cut_off = torch.take_along_dim(
        sorted_confidence, num_to_mask, axis=-1
    )
    logging.debug(f"cut off shape: {cut_off.shape}")

    # mask out the tokens
    mask = confidence < cut_off
    logging.debug(f"mask shape: {mask.shape}")

    return mask

@torch.jit.script_if_tracing
def typical_filter(
        logits: Tensor, 
        typical_mass: float = 0.95,
        typical_min_tokens: int = 1,):
    nb, nt, _ = logits.shape

    # x_flat = rearrange(logits, "b t l -> (b t ) l")
    nb, nt, nl = logits.shape
    x_flat = logits.view(nb * nt, nl)

    x_flat_norm = torch.nn.functional.log_softmax(x_flat, dim=-1)
    x_flat_norm_p = torch.exp(x_flat_norm)
    entropy = -(x_flat_norm * x_flat_norm_p).nansum(-1, keepdim=True)

    c_flat_shifted = torch.abs((-x_flat_norm) - entropy)
    c_flat_sorted, x_flat_indices = torch.sort(c_flat_shifted, descending=False)
    x_flat_cumsum = (
        x_flat.gather(-1, x_flat_indices).softmax(dim=-1).cumsum(dim=-1)
    )

    last_ind = (x_flat_cumsum < typical_mass).sum(dim=-1)
    sorted_indices_to_remove = c_flat_sorted > c_flat_sorted.gather(
        1, last_ind.view(-1, 1)
    )
    if typical_min_tokens > 1:
        sorted_indices_to_remove[..., :typical_min_tokens] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, x_flat_indices, sorted_indices_to_remove
    )
    x_flat = x_flat.masked_fill(indices_to_remove, -float("Inf"))

    # logits = rearrange(x_flat, "(b t) l -> b t l", t=nt)
    logits = x_flat.view(nb, nt, nl)

    return logits


if __name__ == "__main__":
    # import argbind
    from .layers import num_params

    VampNet = argbind.bind(VampNet)

    @argbind.bind(without_prefix=True)
    def try_model(device: str = "cuda", batch_size: int = 2, seq_len_s: float = 10.0):
        seq_len = int(32000 / 512 * seq_len_s)

        model = VampNet().to(device)

        z = torch.randint(
            0, model.vocab_size, size=(batch_size, model.n_codebooks, seq_len)
        ).to(device)

        r = torch.zeros(batch_size).to(device)
        
        z_mask_latent = torch.rand(
            batch_size, model.latent_dim * model.n_codebooks, seq_len
        ).to(device)
        z_hat = model(z_mask_latent)

        pred = z_hat.argmax(dim=1)
        pred = model.embedding.unflatten(pred, n_codebooks=model.n_predict_codebooks)

        logging.debug(f"model has {num_params(model)/1e6:<.3f}M parameters")
        logging.debug(f"prediction has shape {pred.shape}")

    args = argbind.parse_args()
    with argbind.scope(args):
        try_model()


