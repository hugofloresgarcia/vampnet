import math
import logging
from typing import Optional, Tuple, Union, List
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import lightning as L
from einops import rearrange

from x_transformers import TransformerWrapper
from x_transformers import Encoder

from .activations import get_activation
from .layers import CodebookEmbedding
from .layers import FiLM
from .layers import SequentialWithFiLM
from .layers import WNConv1d
from ..util import scalar_to_batch_tensor, codebook_flatten, codebook_unflatten
from ..mask import _gamma, random, stemgen_random

@torch.jit.script_if_tracing
def gumbel_noise_like(t: Tensor):
    noise = torch.zeros_like(t).uniform_(1e-20, 1.0)
    return -torch.log(-torch.log(noise))


def gumbel_sample(t, temperature=1.0, dim=-1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise_like(t)).argmax(dim=dim)

class CodebookEmbedding(nn.Module):
    """ Codebook embedding that is meant to be initialized
    with the codec's RVQ module, to take advantage of it's pre-initialized
    embedding layers
    """
    def __init__(
        self,
        vocab_size: int,
        latent_dim: int,
        n_codebooks: int,
        emb_dim: int,
        input_dim: int,
        special_tokens: Optional[Tuple[str]] = None
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
        from vampnet.dac.nn.quantize import ResidualVectorQuantize
        self.quantizer = ResidualVectorQuantize(
            input_dim=input_dim,
            n_codebooks=n_codebooks,
            codebook_size=vocab_size,
            codebook_dim=latent_dim,
        )

    def from_codes(self, codes: torch.Tensor):
        """ 
        get a sequence of continuous embeddings from a sequence of discrete codes. 
        unlike it's counterpart in the original VQ-VAE, this function adds for any special tokens
        necessary for the language model, like <MASK>. 
        """
        n_codebooks = codes.shape[1]
        latent = []
        for i in range(n_codebooks):
            c = codes[:, i, :]

            lookup_table = self.quantizer.quantizers[i].codebook.weight
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
        """
        project a sequence of latents to a sequence of embeddings
        """
        x = self.out_proj(latents)
        return x

class VampNet(L.LightningModule):
    def __init__(
        self,
        n_heads: int = 12,
        n_layers: int = 12,
        latent_dim: int = 8,
        n_codebooks: int = 9,
        n_conditioning_codebooks: int = 0,
        embedding_dim: int = 1026,
        vocab_size: int = 1024,
        flash_attn: bool = True,
        dropout: float = 0.0, 
        mode: str = "vampnet"
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.n_codebooks = n_codebooks
        self.n_conditioning_codebooks = n_conditioning_codebooks
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.flash_attn = flash_attn

        self.mode = mode
        assert self.mode in ["vampnet", "stemgen"], "mode must be vampnet or stemgen"
        if self.mode == "stemgen":
            self.generate = self.stemgen_generate
            self.random_mask = stemgen_random
        elif self.mode == "vampnet":
            self.generate = self.generate
            self.random_mask = random

        print(("~" * 80 + "\n") * 4)
        print(f"VampNet mode: {self.mode}")
        print(("~" * 80 + "\n") * 4)

        self.embedding = CodebookEmbedding(
            latent_dim=latent_dim,
            n_codebooks=n_codebooks,
            vocab_size=vocab_size,
            emb_dim=embedding_dim,
            input_dim=vocab_size,
            special_tokens=["MASK"],
        )
        self.mask_token = self.embedding.special_idxs["MASK"]

        self.lm = TransformerWrapper(
            num_tokens=self.embedding_dim,
            max_seq_len=2048,
            token_emb=nn.Identity(),
            attn_layers=Encoder(
                dim=self.embedding_dim,
                depth=self.n_layers,
                heads=self.n_heads,
                attn_flash=self.flash_attn,
                ff_glu=True, 
                use_rmsnorm=True, 
                # attn_num_mem_kv = 16,
                rotary_pos_emb=True, # v100 
                # rotary_xpos=True, # new in v101
            ),
            use_abs_pos_emb=False, 
            emb_dropout=dropout,
        )


        # Add final conv layer
        self.n_predict_codebooks = n_codebooks - n_conditioning_codebooks

        # one classifier head per codebook
        self.classifiers = nn.ModuleList([
            WNConv1d(
                embedding_dim,
                vocab_size,
                kernel_size=1,
                padding="same",
            ) for _ in range(self.n_predict_codebooks)
        ])

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
        # # input should be shape (batch, codebook, seq)
        # x = self.codebook_idx_to_global_idx(x)
        # # pass through the embedding layer, output shape (batch, codebook, seq, emb)
        # x = self.embedding(x)
        # # concat the embds along the codebook dimension
        # x = rearrange(x, "b c n d -> b n (c d)")
        # # sum the embds along the codebook dimension
        # x = x.sum(dim=1)

        x = self.embedding(self.embedding.from_codes(x))
        x = rearrange(x, "b n d -> b d n")

        x_mask = torch.ones_like(x, dtype=torch.bool)[:, :, :1].squeeze(2)

        assert return_activations == False, "return_activations not supported sry :( would happily implement if you need it"
        out = self.lm(
            x, return_mems=False, 
            mask=x_mask, 
        )

        out = rearrange(out, "b n d -> b d n")

        # run through each classifier
        # (b n (c d)) -> (b n
        out = torch.stack([
            classifier(out) for classifier in self.classifiers
        ], dim=1) # b, c, t, p

        # b, pc, t = out.shape
        out = rearrange(out, "b c p t -> b p (t c)")

        if return_activations:
            return out, activations
        else:
            return out


    @dataclass
    class GenerationState:
        z_masked: torch.Tensor
        z: torch.Tensor
        mask: torch.Tensor
        num_mask_tokens_at_start: int
        n_infer_codebooks: int
        step: int
        sampling_steps: int
        cfg_guidance: Optional[float]
        nb: int

    @torch.inference_mode()
    def initialize_state(
            self, 
            codes: torch.Tensor, 
            sampling_steps: int, 
            cfg_guidance: Optional[float]
        ) -> GenerationState:
        z = codes
        nb = z.shape[0]

        mask = z == self.mask_token
        z_masked = z.masked_fill(mask.bool(), self.mask_token)
        num_mask_tokens_at_start = (z_masked == self.mask_token).sum().item()
        n_infer_codebooks = self.n_codebooks - self.n_conditioning_codebooks

        if cfg_guidance is not None:
            z_uncond = torch.full_like(z, self.mask_token)
            z_masked = torch.cat((z_masked, z_uncond), dim=0)
            z = torch.cat((z, z_uncond), dim=0)
            mask = torch.cat((mask, torch.full_like(mask, 1)), dim=0)

        return self.GenerationState(
            z_masked=z_masked,
            z=z,
            mask=mask,
            num_mask_tokens_at_start=num_mask_tokens_at_start,
            n_infer_codebooks=n_infer_codebooks,
            step=0,
            sampling_steps=sampling_steps,
            cfg_guidance=cfg_guidance,
            nb=nb,
        )

    @torch.inference_mode()
    def generate_step(self, 
            state: GenerationState, 
            temperature: float, 
            mask_temperature: float, 
            typical_filtering: bool, 
            typical_mass: float, 
            typical_min_tokens: int, 
            top_p: Optional[float], 
            sample_cutoff: float
        ):
        step = state.step
        z_masked, z, mask = state.z_masked, state.z, state.mask
        num_mask_tokens_at_start = state.num_mask_tokens_at_start
        n_infer_codebooks = state.n_infer_codebooks
        cfg_guidance = state.cfg_guidance
        nb = state.nb
        sampling_steps = state.sampling_steps

        # Schedule step
        r = scalar_to_batch_tensor(
            (step + 1) / sampling_steps, z.shape[0]
        ).to(z.device)

        # Forward pass
        logits = self.forward(z_masked)

        if cfg_guidance is not None:
            logits_cond, logits_uncond = logits[:nb], logits[nb:]
            logits = cfg_guidance * logits_cond + (1 - cfg_guidance) * logits_uncond

        logits = logits.permute(0, 2, 1)  # b, seq, prob
        sampled_z, selected_probs = sample_from_logits(
            logits,
            sample=((step / sampling_steps) <= sample_cutoff),
            temperature=temperature,
            typical_filtering=typical_filtering,
            typical_mass=typical_mass,
            typical_min_tokens=typical_min_tokens,
            top_k=None,
            top_p=top_p,
            return_probs=True,
        )

        z_masked = codebook_flatten(z_masked[:, self.n_conditioning_codebooks:, :])
        mask = (z_masked == self.mask_token).int()

        sampled_z = torch.where(mask.bool(), sampled_z, z_masked)
        selected_probs = torch.where(mask.bool(), selected_probs, torch.inf)

        num_to_mask = torch.floor(_gamma(r) * num_mask_tokens_at_start).unsqueeze(1).long()

        if step != (sampling_steps - 1):
            num_to_mask = torch.maximum(
                torch.tensor(1),
                torch.minimum(mask.sum(dim=-1, keepdim=True) - 1, num_to_mask),
            )

        mask = mask_by_random_topk(num_to_mask, selected_probs, mask_temperature * (1 - r))
        z_masked = torch.where(mask.bool(), self.mask_token, sampled_z)

        z_masked = codebook_unflatten(z_masked, n_infer_codebooks)
        mask = codebook_unflatten(mask, n_infer_codebooks)
        z_masked = torch.cat((z[:, :self.n_conditioning_codebooks, :], z_masked), dim=1)

        state.z_masked = z_masked
        state.mask = mask
        state.step += 1

        return state

    @torch.inference_mode()
    def generate(self, 
            codes: torch.Tensor = None, 
            sampling_steps: int = 12, 
            temperature: float = 1.0, 
            mask_temperature: float = 10.5, 
            typical_filtering=False, 
            typical_mass=0.15, 
            typical_min_tokens=64, 
            top_p=None, 
            seed: int = None, 
            sample_cutoff: float = 1.0, 
            causal_weight: float = 0.0, 
            cfg_guidance: float = None
        ):
        from tqdm import tqdm
        if seed is not None:
            torch.manual_seed(seed)

        state = self.initialize_state(codes, sampling_steps, cfg_guidance)

        for _ in tqdm(range(sampling_steps)):
            state = self.generate_step(
                state,
                temperature,
                mask_temperature,
                typical_filtering,
                typical_mass,
                typical_min_tokens,
                top_p,
                sample_cutoff
            )

        return state.z_masked[:state.nb] if cfg_guidance is not None else state.z_masked

    @torch.inference_mode()
    def stemgen_generate(
        self,
        codes: torch.Tensor,
        sampling_steps: List[int] = [16, 8, 8, 2, 2, 2, 2, 1, 1],
        temperature: float = 1.0,
        mask_temperature: float = 10.5,
        random_remask: bool = False,
        typical_filtering=False,
        typical_mass=0.2,
        typical_min_tokens=1,
        top_p=None,
        sample_cutoff: float = 1.0,
        debug=False,
        causal_weight: float = 0.0,
    ):

        assert self.n_conditioning_codebooks == 0, "n_conditioning codebooks is an old vampnet construct"
        #####################
        # resolve initial z #
        #####################
        z = codes

        #################
        # resolve mask #
        #################
        # get the mask from the z
        mask = z == self.mask_token
        orig_mask = mask.clone()

        ###########
        # set up #
        ##########
        # apply the mask to z
        z_masked = z.clone()

        # how many codebooks are we inferring vs conditioning on?
        n_infer_codebooks = self.n_codebooks - self.n_conditioning_codebooks

        #################
        # begin sampling #
        #################
        # add one sampling step for each codebook level
        logging.debug(f"initial mask: {mask}")
        logging.debug(f"adding {n_infer_codebooks} sampling steps")

        # append if we have too little, truncate if we have too many
        steps = sampling_steps + [1 for _ in range(n_infer_codebooks - len(sampling_steps))]
        steps = steps[:n_infer_codebooks]

        for codebook_level, nsteps in enumerate(steps):

            # apply the orig mask to z_masked, only in the current codebook level
            # this is crucial due to the stemgen random masking we did during training
            # which ensures all upper codebooks are masked while inferring the bottom ones.
            z_masked[:, codebook_level, :] = torch.where(
                orig_mask[:, codebook_level, :].bool(), 
                self.mask_token, 
                z[:, codebook_level, :]
            )

            # mask everything above the current codebook level
            z_masked[:, codebook_level+1:, :] = self.mask_token

            # assert that any codebook below us is fully unmasked
            if codebook_level > 0:
                assert (z_masked[:, codebook_level-1, :] != self.mask_token).all()

            # how many mask tokens to begin with?
            num_mask_tokens_at_start = (z_masked[:, codebook_level, :] == self.mask_token).sum(dim=-1)
            logging.debug(f"num mask tokens at start: {num_mask_tokens_at_start}")

            for i in range(nsteps):
                logging.debug(f"processing cb level {codebook_level} of {len(steps)}")
                logging.debug(f"step {i} of {nsteps}")

                # our current schedule step
                r = scalar_to_batch_tensor(
                    (i + 1) / nsteps, 
                    z.shape[0]
                ).to(z.device)
                logging.debug(f"r: {r}")

                # get latents
                logging.debug("z_masked before forward", z_masked)
                _debug_z_masked_before_forward = z_masked.clone()

                logits = self.forward(z_masked)  # b, prob, seq
                logits = rearrange(logits, "b p t -> b t p")
                # print(f"permuted logits with shape: {logits.shape}")

                sampled_z, selected_probs = sample_from_logits(
                    logits, sample=(
                        (i / nsteps) <= sample_cutoff
                    ), 
                    temperature=temperature,
                    typical_filtering=typical_filtering, typical_mass=typical_mass,
                    typical_min_tokens=typical_min_tokens,
                    top_k=None, top_p=top_p, return_probs=True,
                )

                # fill selected probs with -inf if we're not in the codebook level we are sampling from
                # find out which codebook we are sampling from
                selected_probs = codebook_unflatten(selected_probs, n_infer_codebooks)
                selected_probs[:,  codebook_level+1:, :,] = -float("inf") # all the ones above
                logging.debug(f"selected probs: {selected_probs}")
                logging.debug(mask)
                selected_probs = codebook_flatten(selected_probs)

                # print(f"sampled z with shape: {sampled_z.shape}")

                # flatten z_masked and mask, so we can deal with the sampling logic
                # we'll unflatten them at the end of the loop for the next forward pass
                # remove conditioning codebooks, we'll add them back at the end
                z_masked = codebook_flatten(z_masked[:, self.n_conditioning_codebooks:, :])      

                mask = (z_masked == self.mask_token).int()
                logging.debug(f"mask now: {codebook_unflatten(mask, n_infer_codebooks)}")
                
                # update the mask, remove conditioning codebooks from the mask
                logging.debug(f"updated mask with shape: {mask.shape}")
                
                # add z back into sampled z where the mask was false
                sampled_z = torch.where(
                    mask.bool(), sampled_z, z_masked
                )
                logging.debug(f"added z back into sampled z with shape: {sampled_z.shape}")

                # get the num tokens to mask, according to the schedule
                num_to_mask = torch.floor(_gamma(r) * num_mask_tokens_at_start).unsqueeze(1).long()

                # num_to_mask = torch.floor(r * num_mask_tokens_at_start).unsqueeze(1).long() # doesn't work at all this way
                logging.debug(f"num to mask: {num_to_mask}")
                logging.debug(f"masking {num_to_mask.sum()} tokens")

                # mask at least 1 token if we still have steps left? 
                if i != (nsteps - 1):
                    # TODO: hmm why does this make sense?  
                    mask = codebook_unflatten(mask, n_infer_codebooks)
                    num_to_mask = torch.maximum( 
                        torch.tensor(1),
                        torch.minimum(
                            mask[:, codebook_level, :].sum(dim=-1, keepdim=True) - 1,
                            num_to_mask
                        )
                    )
                    logging.debug(f"will mask {num_to_mask.sum()} tokens")
                    mask = codebook_flatten(mask)
            
                # ignore any tokens that weren't masked (inf prob so that they definitely get kept)
                selected_probs = torch.where(
                   (mask.bool()), selected_probs, torch.inf
                )

                # add a causal weight to the selected probs
                causal_probs = torch.linspace(1, 0, z_masked.shape[-1], device=z_masked.device)
                causal_probs = causal_probs.repeat(z_masked.shape[0], 1)
                selected_probs = selected_probs + causal_probs * causal_weight

                # # get our new mask
                ############
                mask = codebook_unflatten(mask, n_infer_codebooks)
                selected_probs = codebook_unflatten(selected_probs, n_infer_codebooks)

                # only consider probs at current level
                selected_probs_cur_level = selected_probs[:, codebook_level, :]

                # TODO: debugging this bad boy rn
                if random_remask:
                    mask_cur_level = mask_by_random(
                        num_to_mask, selected_probs_cur_level
                    )
                else:
                    mask_cur_level = mask_by_random_topk(
                        num_to_mask, selected_probs_cur_level, mask_temperature * (1-r)
                    )  

                mask[:, codebook_level, :] = mask_cur_level

                mask = codebook_flatten(mask)
                selected_probs = codebook_flatten(selected_probs)
                ###############


                # update the mask
                z_masked = torch.where(
                    mask.bool(), self.mask_token, sampled_z
                )
                logging.debug(f"updated z_masked with shape: {z_masked.shape}")

                z_masked = codebook_unflatten(z_masked, n_infer_codebooks)
                mask = codebook_unflatten(mask, n_infer_codebooks)
                logging.debug(f"unflattened z_masked with shape: {z_masked.shape}")

                # add conditioning codebooks back to z_masked
                z_masked = torch.cat(
                    (z[:, :self.n_conditioning_codebooks, :], z_masked), dim=1
                )
                logging.debug(f"added conditioning codebooks back to z_masked with shape: {z_masked.shape}")
                logging.debug(f"\n\n\n")

                if debug:
                    import matplotlib.pyplot as plt
                    from pathlib import Path
                    Path(".vampnet").mkdir(exist_ok=True)
                    plt.clf()
                    # close all figs
                    plt.close('all')
                    # set the fig size
                    plt.subplot(4, 1, 1)
                    # sig =  self.to_signal(sampled_z, codec)
                    # sig.cpu().specshow()
                    # save the orig mask
                    plt.imshow(orig_mask[0].cpu().numpy(), aspect='auto', origin='lower', cmap='gray_r',interpolation='none')

                    plt.subplot(4, 1, 2)       
                    # since z_masked is a codebook, we want to plot the colormap
                    # with distinct colors for each codebook index
                    # plt.imshow(_debug_z_masked_before_forward[0].cpu().numpy(), aspect='auto', origin='lower', cmap="tab20")
                    # make it so that anywhere where the mask is 1, we make that pixel black
                    plt.imshow(_debug_z_masked_before_forward[0].cpu().numpy(), aspect='auto', origin='lower', cmap='gray_r',interpolation='none')


                    plt.subplot(4, 1, 3)
                    # plot the mask (which is a matrix)
                    plt.imshow(mask[0].cpu().numpy(), aspect='auto', origin='lower', cmap='gray_r')
                    plt.subplot(4, 1, 4)
                    # replace any inf or -inf with 0
                    _selected_probs = torch.where(
                        selected_probs == torch.inf, torch.ones_like(selected_probs), selected_probs
                    )
                    _selected_probs = torch.where(
                        selected_probs == -torch.inf, torch.zeros_like(selected_probs), selected_probs
                    )
                    # fig = plt.gcf()
                    # fig.set_figheight(15)
                    # fig.set_figwidth(15)
                    plt.imshow(codebook_unflatten(_selected_probs, n_infer_codebooks)[0].cpu().numpy(), aspect='auto', origin='lower', cmap="viridis", interpolation='none') 
                    # plt.show()
                    plt.savefig(f".vampnet/c={codebook_level}_{i}.png")
                    plt.close('all')


        # add conditioning codebooks back to sampled_z
        sampled_z = codebook_unflatten(sampled_z, n_infer_codebooks)
        sampled_z = torch.cat(
            (z[:, :self.n_conditioning_codebooks, :], sampled_z), dim=1
        )

        logging.debug(f"finished sampling")
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
    
def mask_by_random(
    num_to_mask: int, 
    probs: torch.Tensor,
):
    # breakpoint()
    # check which ones we CAN'T mask by looking for p == inf
    cant_mask = probs == torch.inf

    # now, pick random tokens to mask from the indices we can mask
    can_mask_indices = torch.where(~cant_mask)
    num_to_mask = min(num_to_mask, can_mask_indices[0].shape[0])
    mask_indices = torch.randperm(can_mask_indices[0].shape[0])[:num_to_mask]

    # make a mask from the indices
    mask = torch.zeros_like(probs, dtype=torch.bool)
    mask[can_mask_indices[0][mask_indices], can_mask_indices[1][mask_indices]] = True

    # count how many tokens we masked
    num_masked = mask.sum().item()
    assert num_masked == num_to_mask, f"num masked: {num_masked}, num to mask: {num_to_mask}"

    return mask



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


