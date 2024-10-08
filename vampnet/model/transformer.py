import logging
import math
from typing import Optional
from typing import Tuple
from typing import Union
from typing import List

import audiotools as at
import loralib as lora
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from x_transformers import ContinuousTransformerWrapper
from x_transformers import Encoder
import wandb


import vampnet
from vampnet.mask import (
    _gamma, scalar_to_batch_tensor, full_mask, empty_mask
)
from vampnet.util import (
    codebook_flatten, codebook_unflatten
)
from vampnet.model.layers import CodebookEmbedding, WNConv1d


from ..mask import _gamma, scalar_to_batch_tensor, full_mask
from ..util import codebook_flatten
from ..util import codebook_unflatten
from .layers import CodebookEmbedding
from .layers import WNConv1d

# set the logging level to info
logging.basicConfig(level=logging.INFO)


class ClassifierFreeGuidanceDropout(nn.Module):

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        if self.training and self.dropout > 0:
            mask = torch.rand(x.shape[0], 1, 1, device=x.device) > self.dropout
        else: 
            mask = torch.ones(x.shape[0], 1, 1, device=x.device)    
        
        return mask.bool()

class VampNet(at.ml.BaseModel):
    def __init__(
        self,
        n_heads: int = vampnet.N_HEADS,
        n_layers: int = vampnet.N_LAYERS,
        n_codebooks: int = vampnet.N_CODEBOOKS,
        n_conditioning_codebooks: int = vampnet.N_CONDITIONING_CODEBOOKS,
        latent_dim: int = vampnet.LATENT_DIM,
        embedding_dim: int = vampnet.EMBEDDING_DIM,
        vocab_size: int = vampnet.VOCAB_SIZE,
        dropout: float = vampnet.DROPOUT,
        d_ctrl: int = vampnet.D_CTRL,
        cfg_dropout: float = 0.2,
        max_seq_len: int = vampnet.MAX_SEQ_LEN,
        num_reg_tokens: int = vampnet.NUM_REG_TOKENS,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.n_codebooks = n_codebooks
        self.n_conditioning_codebooks = n_conditioning_codebooks
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.max_seq_len = max_seq_len
        self.num_reg_tokens = num_reg_tokens
        self.d_ctrl = d_ctrl
        self.dropout = dropout
        self.cfg_dropout = cfg_dropout

        self.embedding = CodebookEmbedding(
            latent_dim=latent_dim,
            n_codebooks=n_codebooks,
            vocab_size=vocab_size,
            emb_dim=embedding_dim,
            special_tokens=["MASK"],
        )
        self.special_tokens = self.embedding.special_idxs
        self.mask_token = self.special_tokens["MASK"]

        self.lm = ContinuousTransformerWrapper(
            max_seq_len=max_seq_len,
            attn_layers=Encoder(
                dim=self.embedding_dim,
                depth=self.n_layers,
                heads=self.n_heads,
                attn_flash=True,
                ff_glu=True, 
                use_rmsnorm=True, 
                rotary_pos_emb=True
            ),
            emb_dropout=dropout,
            num_memory_tokens=num_reg_tokens,
        )

        if d_ctrl > 0:
            self.ctrl_proj = lora.Linear(
                self.d_ctrl,
                self.embedding_dim,
                r=vampnet.LORA_R,
            )
            self.ctrl_dropout = ClassifierFreeGuidanceDropout(cfg_dropout)
            # self.ctrl_mask_emb = nn.Parameter(torch.randn(1, 1, self.embedding_dim), requires_grad=True)

        # Add final conv layer
        self.n_predict_codebooks = n_codebooks - n_conditioning_codebooks
        self.classifier = lora.Linear(
                embedding_dim,
                vocab_size * self.n_predict_codebooks,
        )



    def forward(self, x, pad_mask=None, ctrl=None, ctrl_mask=None):
        if ctrl_mask is None and ctrl is not None:
            ctrl_mask = torch.ones_like(ctrl[:, 0, :]).bool()
        if pad_mask is None:
            pad_mask = torch.ones_like(x[:, 0, :]).bool()

        pad_mask = pad_mask.bool() if isinstance(pad_mask, torch.Tensor) else pad_mask
        ctrl_mask = ctrl_mask.bool() if isinstance(ctrl_mask, torch.Tensor) else ctrl_mask
        x = self.embedding(x)
        x = rearrange(x, "b d n -> b n d")

        if self.d_ctrl > 0:
            assert ctrl is not None, "ctrl is required if d_ctrl > 0"
            ctrl = rearrange(ctrl, "b d n -> b n d")
            ctrl_mask = rearrange(ctrl_mask, "b n -> b n 1")

            # apply the control projection
            ctrl = self.ctrl_proj(ctrl)

            # replace the control with the mask embedding where the mask is true

            # apply dropout to the control
            ctrl_dropout_mask = self.ctrl_dropout(ctrl)
            ctrl = torch.where(ctrl_mask, ctrl, torch.zeros_like(ctrl))
            ctrl = torch.where(ctrl_dropout_mask, ctrl, torch.zeros_like(ctrl))

            # add the control to the input
            x = x + ctrl

        out = self.lm(
            x, return_mems=False, 
            mask=pad_mask, 
        )
        out = self.classifier(out)
        out = rearrange(out, "b n d -> b d n")
        out = rearrange(out, "b (p c) t -> b p (t c)", c=self.n_predict_codebooks)

        return out


    @torch.inference_mode()
    def generate(
        self, 
        codec,
        time_steps: int = 300,
        _sampling_steps: List[int] = [24],
        start_tokens: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        mask: Optional[torch.Tensor] = None,
        mask_temperature: float = 10.5,
        typical_filtering=True,
        typical_mass=0.2,
        typical_min_tokens=64,
        top_p=None,
        seed: int = None,
        sample_cutoff: float = 1.0,
        causal_weight: float = 0.0,
        cfg_guidance: Optional[float] = None,
    ):
        return_signal = False
        debug=False
        
        if seed is not None:
            at.util.seed(seed)
        sampling_steps = sum(_sampling_steps)
        logging.debug(f"beginning generation with {sampling_steps} steps")

        ##################### 
        # resolve initial z #
        #####################
        z = start_tokens
        nb = z.shape[0]

        if z is None:
            z = torch.full((1, self.n_codebooks, time_steps), self.mask_token).to(
                self.device
            )

        #################
        # resolve mask #
        #################

        if mask is None:
            mask = torch.ones_like(z).to(self.device).int()
            mask[:, : self.n_conditioning_codebooks, :] = 0.0
        if mask.ndim == 2:
            mask = mask[:, None, :].repeat(1, z.shape[1], 1)
        # init_mask = mask.clone()
    
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

            # get latents
            latents = self.embedding.from_codes(z_masked, codec)


            # infer from latents
            # NOTE: this collapses the codebook dimension into the sequence dimension
            logits = self.forward(latents) # b, prob, seq

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

        if return_signal:
            return self.to_signal(sampled_z, codec)
        else:
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

    if typical_filtering and sample:
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
        v, sorted_indices = logits.sort(dim=-1, descending=True)
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
    confidence = torch.log(probs) + temperature * noise
    logging.debug(f"confidence shape: {confidence.shape}")

    sorted_confidence, sorted_idx = confidence.sort(dim=-1)
    logging.debug(f"sorted confidence shape: {sorted_confidence.shape}")
    logging.debug(f"sorted idx shape: {sorted_idx.shape}")

    # get the cut off threshold, given the mask length
    cut_off = torch.take_along_dim(sorted_confidence, num_to_mask, axis=-1)
    logging.debug(f"cut off shape: {cut_off.shape}")

    # mask out the tokens
    mask = confidence < cut_off
    logging.debug(f"mask shape: {mask.shape}")

    return mask


def typical_filter(
    logits,
    typical_mass: float = 0.95,
    typical_min_tokens: int = 1,
):
    nb, nt, _ = logits.shape
    x_flat = rearrange(logits, "b t l -> (b t ) l")
    x_flat_norm = torch.nn.functional.log_softmax(x_flat, dim=-1)
    x_flat_norm_p = torch.exp(x_flat_norm)
    entropy = -(x_flat_norm * x_flat_norm_p).nansum(-1, keepdim=True)

    c_flat_shifted = torch.abs((-x_flat_norm) - entropy)
    c_flat_sorted, x_flat_indices = torch.sort(c_flat_shifted, descending=False)
    x_flat_cumsum = x_flat.gather(-1, x_flat_indices).softmax(dim=-1).cumsum(dim=-1)

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
    logits = rearrange(x_flat, "(b t) l -> b t l", t=nt)
    return logits


def gumbel_noise_like(t):
    noise = torch.zeros_like(t).uniform_(1e-20, 1)
    return -torch.log(-torch.log(noise))


def gumbel_sample(t, temperature=1.0, dim=-1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise_like(t)).argmax(dim=dim)


# if __name__ == "__main__":
#     import argbind
#     # from .layers import num_params

#     VampNet = argbind.bind(VampNet)

#     @argbind.bind(without_prefix=True)
#     def try_model(device: str = "cuda", batch_size: int = 2, seq_len_s: float = 10.0):
#         seq_len = int(32000 / 512 * seq_len_s)

#         model = VampNet().to(device)

#         z = torch.randint(
#             0, model.vocab_size, size=(batch_size, model.n_codebooks, seq_len)
#         ).to(device)

#         r = torch.zeros(batch_size).to(device)

#         z_mask_latent = torch.rand(
#             batch_size, model.latent_dim * model.n_codebooks, seq_len
#         ).to(device)
#         z_hat = model(z_mask_latent)

#         pred = z_hat.argmax(dim=1)
#         pred = codebook_unflatten(pred, n_c=model.n_predict_codebooks)

#         print(f"model has {vampnet.util.num_params(model)/1e6:<.3f}M parameters")
#         print(f"prediction has shape {pred.shape}")
#         breakpoint()

#     args = argbind.parse_args()
#     with argbind.scope(args):
#         try_model()

