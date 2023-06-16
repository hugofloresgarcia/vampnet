import math
import logging
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import loralib as lora
import audiotools as at

from .activations import get_activation
from .layers import CodebookEmbedding
from .layers import FiLM
from .layers import SequentialWithFiLM
from .layers import WNConv1d
from ..util import scalar_to_batch_tensor, codebook_flatten, codebook_unflatten
from ..mask import _gamma

LORA_R = 8

# def log(t, eps=1e-20):
#     return torch.log(t + eps)


def gumbel_noise_like(t):
    noise = torch.zeros_like(t).uniform_(1e-20, 1)
    return -torch.log(-torch.log(noise))


def gumbel_sample(t, temperature=1.0, dim=-1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise_like(t)).argmax(dim=dim)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.var_eps = eps

    def forward(self, x):
        """Returns root mean square normalized version of input `x`
        # T5 uses a layer_norm which only scales and doesn't shift, which is also known
        # as Root Mean Square Layer Normalization https://arxiv.org/abs/1910.07467
        # thus varience is calculated w/o mean and there is no bias
        Parameters
        ----------
        x : Tensor[B x T x D]
        Returns
        -------
        Tensor[B x T x D]
        """
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.var_eps)

        return self.weight * x


class FeedForward(nn.Module):
    def __init__(
        self, d_model: int = 512, dropout: float = 0.1, activation: str = "geglu"
    ):
        super().__init__()
        factor = 2 if activation == "geglu" else 1
        self.w_1 = lora.Linear(d_model, d_model * 4, bias=False, r=LORA_R)
        self.w_2 = lora.Linear(d_model * 4 // factor, d_model, bias=False, r=LORA_R)
        self.drop = nn.Dropout(dropout)
        self.act = get_activation(activation)()

    def forward(self, x):
        """Computes position-wise feed-forward layer
        Parameters
        ----------
        x : Tensor[B x T x D]
        Returns
        -------
        Tensor[B x T x D]
        """
        x = self.w_1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.w_2(x)
        return x


class MultiHeadRelativeAttention(nn.Module):
    def __init__(
        self,
        n_head: int = 8,
        d_model: int = 512,
        dropout: float = 0.1,
        bidirectional: bool = True,
        has_relative_attention_bias: bool = True,
        attention_num_buckets: int = 32,
        attention_max_distance: int = 128,
    ):
        super().__init__()
        d_head = d_model // n_head
        self.n_head = n_head
        self.d_head = d_head
        self.bidirectional = bidirectional
        self.has_relative_attention_bias = has_relative_attention_bias
        self.attention_num_buckets = attention_num_buckets
        self.attention_max_distance = attention_max_distance

        # Create linear query, key, value projections
        self.w_qs = lora.Linear(d_model, d_model, bias=False, r=LORA_R)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = lora.Linear(d_model, d_model, bias=False, r=LORA_R)

        # Create linear final output projection
        self.fc = lora.Linear(d_model, d_model, bias=False, r=LORA_R)

        # Dropout for attention output weights
        self.dropout = nn.Dropout(dropout)

        # Create relative positional embeddings (if turned on)
        if has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(attention_num_buckets, n_head)

    def _relative_position_bucket(self, relative_position):
        """Converts unbounded relative position into bounded set of buckets
        with half "exact" buckets (1 position = 1 bucket) and half "log-spaced"
        buckets
        Parameters
        ----------
        relative_position : Tensor[T_q x T_kv]
            Relative positions between queries and key_value items
        Returns
        -------
        Tensor[T_q x T_kv]
            Input relative positions converted into buckets
        """
        relative_buckets = 0
        num_buckets = self.attention_num_buckets
        max_distance = self.attention_max_distance

        # Convert relative position for (-inf, inf) to [0, inf]
        # Negative relative positions correspond to past
        # Positive relative positions correspond to future
        if self.bidirectional:
            # use half buckets for each side (past / future)
            num_buckets //= 2

            # Shift the position positions by `num_buckets` to wrap around
            # negative positions
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            # If not bidirectional, ignore positive positions and wrap
            # negative positions to positive
            relative_position = -torch.min(
                relative_position, torch.zeros_like(relative_position)
            )

        # Allocate half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in
        # positions up to `max_distance`
        relative_postion_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)

        # Clip the max relative position to `num_buckets - 1`
        relative_postion_if_large = torch.min(
            relative_postion_if_large,
            torch.full_like(relative_postion_if_large, num_buckets - 1),
        )

        # Choose relative buckets based on small or large positions
        relative_buckets += torch.where(
            is_small, relative_position, relative_postion_if_large
        )

        return relative_buckets

    def compute_bias(self, query_length, key_length):
        """Computes a position bias scalar for each index in query_length x key_length
        Parameters
        ----------
        query_length : int
        key_length : int
        Returns
        -------
        Tensor[heads x 1 x T_q x T_kv]
            Position bias to be applied on attention logits
        """

        query_position = torch.arange(query_length, dtype=torch.long)[:, None]
        key_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = key_position - query_position

        # Convert relative position to buckets
        relative_position_bucket = self._relative_position_bucket(relative_position)
        relative_position_bucket = relative_position_bucket.to(
            self.relative_attention_bias.weight.device
        )

        # Index attention bias values
        values = self.relative_attention_bias(relative_position_bucket)
        values = rearrange(values, "q k h -> h 1 q k")

        return values

    def forward(self, q, k, v, mask=None, position_bias=None):
        """Computes attention over (keys, values) for every timestep in query
        Parameters
        ----------
        q : Tensor[B x T_q x d_model]
            Query vectors
        k : Tensor[B x T_kv x d_model]
            Key vectors to compute attention over
        v : Tensor[B x T_kv x d_model]
            Value vectors corresponding to the keys
        mask : Tensor[B x T_q x T_kv], optional
        position_bias: Tensor[head x 1 x T_q x T_kv]
        Returns
        -------
        Tensor[B x T_q x d_model]
            Outputs after attending (key, value) using queries
        """
        # Compute query, key, value projections
        q = rearrange(self.w_qs(q), "b l (head k) -> head b l k", head=self.n_head)
        k = rearrange(self.w_ks(k), "b t (head k) -> head b t k", head=self.n_head)
        v = rearrange(self.w_vs(v), "b t (head k) -> head b t k", head=self.n_head)

        # Compute attention matrix
        attn = torch.einsum("hblk,hbtk->hblt", [q, k]) / np.sqrt(q.shape[-1])

        # Add relative position bias to attention scores
        if position_bias is None:
            if self.has_relative_attention_bias:
                position_bias = self.compute_bias(q.size(-2), k.size(-2))
            else:
                position_bias = torch.zeros_like(attn)
        attn += position_bias

        # Apply mask to attention scores to prevent looking up invalid locations
        if mask is not None:
            attn = attn.masked_fill(mask[None] == 0, -1e9)

        # Normalize attention scores and add dropout
        attn = torch.softmax(attn, dim=3)
        attn = self.dropout(attn)

        # Compute attended outputs (product of attention matrix and values)
        output = torch.einsum("hblt,hbtv->hblv", [attn, v])
        output = rearrange(output, "head b l v -> b l (head v)")
        output = self.fc(output)

        return output, position_bias


class TransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        d_cond: int = 64,
        n_heads: int = 8,
        bidirectional: bool = True,
        is_decoder: bool = False,
        has_relative_attention_bias: bool = False,
        flash_attn: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Store args
        self.is_decoder = is_decoder

        # Create self-attention layer
        self.norm_1 = RMSNorm(d_model)
        self.film_1 = FiLM(d_cond, d_model)
        self.flash_attn = flash_attn

        if flash_attn:
            from flash_attn.flash_attention import FlashMHA
            self.self_attn = FlashMHA(
                embed_dim=d_model,
                num_heads=n_heads,
                attention_dropout=dropout,
                causal=False,
            )
        else:
            self.self_attn = MultiHeadRelativeAttention(
                n_heads, d_model, dropout, bidirectional, has_relative_attention_bias
            )

        # (Optional) Create cross-attention layer
        if is_decoder:
            self.norm_2 = RMSNorm(d_model)
            self.film_2 = FiLM(d_cond, d_model)
            self.cross_attn = MultiHeadRelativeAttention(
                n_heads,
                d_model,
                dropout,
                bidirectional=True,
                has_relative_attention_bias=False,
            )

        # Create last feed-forward layer
        self.norm_3 = RMSNorm(d_model)
        self.film_3 = FiLM(d_cond, d_model)
        self.feed_forward = FeedForward(d_model=d_model, dropout=dropout)

        # Create dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        x_mask,
        cond,
        src=None,
        src_mask=None,
        position_bias=None,
        encoder_decoder_position_bias=None,
    ):
        """Computes one transformer layer consisting of self attention, (op) cross attention
        and feedforward layer
        Parameters
        ----------
        x : Tensor[B x T_q x D]
        x_mask : Tensor[B x T_q]
        src : Tensor[B x T_kv x D], optional
        src_mask : Tensor[B x T_kv x D], optional
        position_bias : Tensor[heads x B x T_q x T_q], optional
            Relative position bias for self attention layer
        encoder_decoder_position_bias : Tensor[heads x B x T_q x T_kv], optional
            Relative position bias for cross attention layer
        Returns
        -------
        Tensor[B x T_q x D]
        """
        y = self.norm_1(x)
        y = self.film_1(y.permute(0, 2, 1), cond).permute(0, 2, 1)
        if self.flash_attn:
            with torch.autocast(y.device.type, dtype=torch.bfloat16):
                y = self.self_attn(y)[0]
        else:
            y, position_bias = self.self_attn(y, y, y, x_mask, position_bias)
        x = x + self.dropout(y)

        if self.is_decoder:
            y = self.norm_2(x)
            y = self.film_2(y.permute(0, 2, 1), cond).permute(0, 2, 1)
            y, encoder_decoder_position_bias = self.cross_attn(
                y, src, src, src_mask, encoder_decoder_position_bias
            )
            x = x + self.dropout(y)

        y = self.norm_3(x)
        y = self.film_3(
            y.permute(
                0,
                2,
                1,
            ),
            cond,
        ).permute(0, 2, 1)
        y = self.feed_forward(y)
        x = x + self.dropout(y)

        return x, position_bias, encoder_decoder_position_bias


class TransformerStack(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        d_cond: int = 64,
        n_heads: int = 8,
        n_layers: int = 8,
        last_layer: bool = True,
        bidirectional: bool = True,
        flash_attn: bool = False,
        is_decoder: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Store args
        self.bidirectional = bidirectional
        self.is_decoder = is_decoder

        # Create transformer layers
        # In T5, relative attention bias is shared by all layers in the stack
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    d_model,
                    d_cond,
                    n_heads,
                    bidirectional,
                    is_decoder,
                    has_relative_attention_bias=True if (i == 0) else False,
                    flash_attn=flash_attn,
                    dropout=dropout,
                )
                for i in range(n_layers)
            ]
        )

        # Perform last normalization
        self.norm = RMSNorm(d_model) if last_layer else None

    def subsequent_mask(self, size):
        return torch.ones(1, size, size).tril().bool()

    def forward(self, x, x_mask, cond=None, src=None, src_mask=None):
        """Computes a full transformer stack
        Parameters
        ----------
        x : Tensor[B x T_q x D]
        x_mask : Tensor[B x T_q]
        src : Tensor[B x T_kv x D], optional
        src_mask : Tensor[B x T_kv], optional
        Returns
        -------
        Tensor[B x T_q x D]
        """

        # Convert `src_mask` to (B x T_q x T_kv) shape for cross attention masking
        if self.is_decoder:
            src_mask = x_mask.unsqueeze(-1) * src_mask.unsqueeze(-2)

        # Convert `x_mask` to (B x T_q x T_q) shape for self attention masking
        x_mask = x_mask.unsqueeze(-2)
        if not self.bidirectional:
            x_mask = x_mask * self.subsequent_mask(x.size(1)).to(x_mask.device)

        # Initialize position biases
        position_bias = None
        encoder_decoder_position_bias = None

        # Compute transformer layers
        for layer in self.layers:
            x, position_bias, encoder_decoder_position_bias = layer(
                x=x,
                x_mask=x_mask,
                cond=cond,
                src=src,
                src_mask=src_mask,
                position_bias=position_bias,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
            )

        return self.norm(x) if self.norm is not None else x


class VampNet(at.ml.BaseModel):
    def __init__(
        self,
        n_heads: int = 20,
        n_layers: int = 16,
        r_cond_dim: int = 64,
        n_codebooks: int = 9,
        n_conditioning_codebooks: int = 0,
        latent_dim: int = 8,
        embedding_dim: int = 1280,
        vocab_size: int = 1024,
        flash_attn: bool = True,
        noise_mode: str = "mask",
        dropout: float = 0.1
    ):
        super().__init__()
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

        self.embedding = CodebookEmbedding(
            latent_dim=latent_dim,
            n_codebooks=n_codebooks,
            vocab_size=vocab_size,
            emb_dim=embedding_dim,
            special_tokens=["MASK"],
        )
        self.mask_token = self.embedding.special_idxs["MASK"]

        self.transformer = TransformerStack(
            d_model=embedding_dim,
            d_cond=r_cond_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            last_layer=True,
            bidirectional=True,
            flash_attn=flash_attn,
            is_decoder=False,
            dropout=dropout,
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

    def forward(self, x, cond):
        x = self.embedding(x)
        x_mask = torch.ones_like(x, dtype=torch.bool)[:, :1, :].squeeze(1)

        cond = self.r_embed(cond)

        x = rearrange(x, "b d n -> b n d")
        out = self.transformer(x=x, x_mask=x_mask, cond=cond)
        out = rearrange(out, "b n d -> b d n")

        out = self.classifier(out, cond)

        out = rearrange(out, "b (p c) t -> b p (t c)", c=self.n_predict_codebooks)

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
    
    @torch.no_grad()
    def to_signal(self, z, codec):
        """
        convert a sequence of latents to a signal. 
        """
        assert z.ndim == 3

        signal = at.AudioSignal(
            codec.decode(
                codec.quantizer.from_latents(self.embedding.from_codes(z, codec))[0]
            )["audio"],
            codec.sample_rate,
        )

        # find where the mask token is and replace it with silence in the audio
        for tstep in range(z.shape[-1]):
            if torch.any(z[:, :, tstep] == self.mask_token):
                sample_idx_0 = tstep * codec.hop_length
                sample_idx_1 = sample_idx_0 + codec.hop_length
                signal.samples[:, :, sample_idx_0:sample_idx_1] = 0.0

        return signal

    def add_truth_to_logits(
        self,
        z_true,
        z_hat,
        mask,
    ):
        z_true = z_true[:, self.n_conditioning_codebooks :, :]
        mask = mask[:, self.n_conditioning_codebooks :, :]

        truth = F.one_hot(z_true, self.vocab_size)
        mask = mask[:, :, :, None].expand(-1, -1, -1, self.vocab_size)
        z_hat = rearrange(
            z_hat,
            "b p (t c) -> b c t p",
            c=self.n_codebooks - self.n_conditioning_codebooks,
        )

        z_hat = z_hat * mask + truth * (1 - mask)

        z_hat = rearrange(z_hat, "b c t p -> b p (t c)")

        return z_hat
    

    @torch.no_grad()
    def sample(
        self,
        codec,
        time_steps: int = 300,
        sampling_steps: int = 36,
        start_tokens: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        temperature: Union[float, Tuple[float, float]] = 0.8,
        top_k: int = None,
        sample: str = "gumbel",
        typical_filtering=True,
        typical_mass=0.2,
        typical_min_tokens=1,
        return_signal=True,
    ):
        if isinstance(temperature, float):
            temperature = torch.tensor(temperature).repeat(sampling_steps)
        elif isinstance(temperature, tuple):
            assert len(temperature) == 2
            l, h = temperature
            temperature = torch.linspace(l, h, sampling_steps)
        else:
            raise TypeError(f"invalid type for temperature")

        z = start_tokens

        if z is None:
            z = torch.full((1, self.n_codebooks, time_steps), self.mask_token).to(
                self.device
            )

        if mask is None:
            mask = torch.ones_like(z).to(self.device).int()
            mask[:, : self.n_conditioning_codebooks, :] = 0.0
        if mask.ndim == 2:
            mask = mask[:, None, :].repeat(1, z.shape[1], 1)

        # figure out which timesteps we're keeping
        keep_mask = 1 - mask

        # any conditioning codebook levels need to be in the keep mask
        # if self.n_conditioning_codebooks > 0:
        #     cond_mask = torch.ones(z.shape[0], self.n_conditioning_codebooks, z.shape[-1]).to(z.device)
        #     keep_mask = torch.cat([cond_mask, keep_mask], dim=1)

        # flatten
        keep_mask = codebook_flatten(keep_mask)

        # our r steps
        r_steps = torch.linspace(0, 1, sampling_steps + 1)[1:].to(self.device)

        # how many tokens did we keep on init?
        num_kept_on_init = keep_mask.sum()

        # how many codebooks are we inferring vs conditioning on?
        n_infer_codebooks = self.n_codebooks - self.n_conditioning_codebooks

        for i in range(sampling_steps):
            # our current temperature
            tmpt = temperature[i]

            # our current schedule step
            r = r_steps[i : i + 1]

            with torch.inference_mode():
                # mask our z
                keep_mask_unflat = codebook_unflatten(keep_mask, n_c=self.n_codebooks)
                z_masked = z.masked_fill(~keep_mask_unflat.bool(), self.mask_token)

                # get latents
                latents = self.embedding.from_codes(z_masked, codec)

                # infer from latents
                logits = self.forward(latents, r)
                logits = logits.permute(0, 2, 1)  # b, seq, prob

                # the schedule determines how many samples to keep
                num_tokens_to_infer = (z.shape[-1] * z.shape[-2]) - num_kept_on_init
                num_to_keep = num_kept_on_init + int(
                    num_tokens_to_infer * (_gamma(1 - r))
                )

                # figure out which logits we wanna keep
                if num_to_keep > 0:
                    probs = logits.softmax(dim=-1)

                    # do mod self.vocab_size to make sure we don't sample from the mask token
                    # in case the mask token was in the og z
                    keep_probs = F.one_hot(z%self.vocab_size, self.vocab_size)[:, :, :]

                    probs = rearrange(
                        probs, "b (t c) p -> b c t p", c=n_infer_codebooks
                    )
                    probs = torch.cat(
                        [keep_probs[:, : self.n_conditioning_codebooks, ...], probs],
                        dim=1,
                    )

                    keep_probs = rearrange(
                        keep_probs, "b c t p -> b (t c) p", c=self.n_codebooks
                    )
                    probs = rearrange(probs, "b c t p -> b (t c) p", c=self.n_codebooks)

                    keep_prob_mask = keep_mask.unsqueeze(-1).repeat(
                        1, 1, self.vocab_size
                    )
                    probs = (keep_prob_mask.long() * keep_probs) + (
                        1 - keep_prob_mask.long()
                    ) * probs

                    highest_probs = probs.max(dim=-1, keepdim=False)[0]
                    v, _ = highest_probs.topk(num_to_keep, dim=-1)

                    keep_mask = torch.ones_like(keep_mask).bool().clone()
                    keep_mask[highest_probs < v[..., [-1]]] = 0

                logits = torch.log(probs)

                z_inferred = sample_from_logits(
                    logits=logits,
                    top_k=top_k,
                    temperature=tmpt,
                    sample=sample,
                    typical_filtering=typical_filtering,
                    typical_mass=typical_mass,
                    typical_min_tokens=typical_min_tokens,
                )

                z = codebook_unflatten(z_inferred, n_c=self.n_codebooks)


        if return_signal:
            return self.to_signal(z, codec)
        else:
            return z

    @torch.no_grad()
    def generate(
        self,
        codec,
        time_steps: int = 300,
        sampling_steps: int = 36,
        start_tokens: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        temperature: Union[float, Tuple[float, float]] = 8.0,
        typical_filtering=False,
        typical_mass=0.2,
        typical_min_tokens=1,
        return_signal=True,
    ):
        logging.info(f"beginning generation with {sampling_steps} steps")

        #####################
        # resolve temperature #
        #####################
        if isinstance(temperature, float):
            temperature = torch.tensor(temperature).repeat(sampling_steps)
        elif isinstance(temperature, tuple):
            assert len(temperature) == 2
            l, h = temperature
            temperature = torch.linspace(l, h, sampling_steps)
        else:
            raise TypeError(f"invalid type for temperature")
        
        logging.info(f"temperature: {temperature}")


        ##################### 
        # resolve initial z #
        #####################
        z = start_tokens

        if z is None:
            z = torch.full((1, self.n_codebooks, time_steps), self.mask_token).to(
                self.device
            )

        logging.info(f"created z with shape {z.shape}")


        #################
        # resolve mask #
        #################

        if mask is None:
            mask = torch.ones_like(z).to(self.device).int()
            mask[:, : self.n_conditioning_codebooks, :] = 0.0
        if mask.ndim == 2:
            mask = mask[:, None, :].repeat(1, z.shape[1], 1)
        # init_mask = mask.clone()
        
        logging.info(f"created mask with shape {mask.shape}")


        ###########
        # set up #
        ##########
        # apply the mask to z
        z_masked = z.masked_fill(mask.bool(), self.mask_token)
        # logging.info(f"z_masked: {z_masked}")

        # how many mask tokens to begin with?
        num_mask_tokens_at_start = (z_masked == self.mask_token).sum()
        logging.info(f"num mask tokens at start: {num_mask_tokens_at_start}")

        # our r steps
        r_steps = torch.linspace(1e-10, 1, sampling_steps+1)[1:].to(self.device)
        logging.info(f"r steps: {r_steps}")

        # how many codebooks are we inferring vs conditioning on?
        n_infer_codebooks = self.n_codebooks - self.n_conditioning_codebooks
        logging.info(f"n infer codebooks: {n_infer_codebooks}")

        #################
        # begin sampling #
        #################

        for i in range(sampling_steps):
            logging.info(f"step {i} of {sampling_steps}")

            # our current temperature
            tmpt = temperature[i]
            logging.info(f"temperature: {tmpt}")

            # our current schedule step
            r = r_steps[i : i + 1]
            logging.info(f"r: {r}")

            # get latents
            latents = self.embedding.from_codes(z_masked, codec)
            logging.info(f"computed latents with shape: {latents.shape}")


            # infer from latents
            # NOTE: this collapses the codebook dimension into the sequence dimension
            logits = self.forward(latents, r) # b, prob, seq
            logits = logits.permute(0, 2, 1)  # b, seq, prob
            if typical_filtering:
                typical_filter(logits, 
                               typical_mass=typical_mass, 
                               typical_min_tokens=typical_min_tokens
                )


            logging.info(f"permuted logits with shape: {logits.shape}")


            # logits2probs
            probs = torch.softmax(logits, dim=-1)
            logging.info(f"computed probs with shape: {probs.shape}")


            # sample from logits with multinomial sampling
            b = probs.shape[0]
            probs = rearrange(probs, "b seq prob -> (b seq) prob")

            sampled_z =  torch.multinomial(probs, 1).squeeze(-1)

            sampled_z = rearrange(sampled_z, "(b seq)-> b seq", b=b)
            probs = rearrange(probs, "(b seq) prob -> b seq prob", b=b)
            logging.info(f"sampled z with shape: {sampled_z.shape}")


            # flatten z_masked and mask, so we can deal with the sampling logic
            # we'll unflatten them at the end of the loop for the next forward pass
            # remove conditioning codebooks, we'll add them back at the end
            z_masked = codebook_flatten(z_masked[:, self.n_conditioning_codebooks:, :])            

            mask = (z_masked == self.mask_token).int()
            
            # update the mask, remove conditioning codebooks from the mask
            logging.info(f"updated mask with shape: {mask.shape}")
            # add z back into sampled z where the mask was false
            sampled_z = torch.where(
                mask.bool(), sampled_z, z_masked
            )
            logging.info(f"added z back into sampled z with shape: {sampled_z.shape}")


            # get the confidences: which tokens did we sample? 
            selected_probs = (
                torch.take_along_dim(
                    probs, sampled_z.long().unsqueeze(-1), 
                    dim=-1
                ).squeeze(-1)
            )

            # ignore any tokens that weren't masked
            selected_probs = torch.where(
                mask.bool(), selected_probs, torch.inf
            )

            # get the num tokens to mask, according to the schedule
            num_to_mask = torch.floor(_gamma(r) * num_mask_tokens_at_start).unsqueeze(1).long()
            logging.info(f"num to mask: {num_to_mask}")

            num_to_mask = torch.maximum(
                torch.tensor(1),
                torch.minimum(
                    mask.sum(dim=-1, keepdim=True) - 1,
                    num_to_mask
                )
            )


            # get our new mask
            mask = mask_by_random_topk(
                num_to_mask, selected_probs, tmpt * (1-r)
            )  

            # update the mask
            z_masked = torch.where(
                mask.bool(), self.mask_token, sampled_z
            )
            logging.info(f"updated z_masked with shape: {z_masked.shape}")

            z_masked = codebook_unflatten(z_masked, n_infer_codebooks)
            mask = codebook_unflatten(mask, n_infer_codebooks)
            logging.info(f"unflattened z_masked with shape: {z_masked.shape}")

            # add conditioning codebooks back to z_masked
            z_masked = torch.cat(
                (z[:, :self.n_conditioning_codebooks, :], z_masked), dim=1
            )
            logging.info(f"added conditioning codebooks back to z_masked with shape: {z_masked.shape}")


        # add conditioning codebooks back to sampled_z
        sampled_z = codebook_unflatten(sampled_z, n_infer_codebooks)
        sampled_z = torch.cat(
            (z[:, :self.n_conditioning_codebooks, :], sampled_z), dim=1
        )

        logging.info(f"finished sampling")

        if return_signal:
            return self.to_signal(sampled_z, codec)
        else:
            return sampled_z


def mask_by_random_topk(num_to_mask: int, probs: torch.Tensor, temperature: float = 1.0):
    """
    Args:
        num_to_mask (int): number of tokens to mask
        probs (torch.Tensor): probabilities for each sampled event, shape (batch, seq)
        temperature (float, optional): temperature. Defaults to 1.0.
    """
    logging.info(f"masking by random topk")
    logging.info(f"num to mask: {num_to_mask}")
    logging.info(f"probs shape: {probs.shape}")
    logging.info(f"temperature: {temperature}")
    logging.info("")

    confidence = torch.log(probs) + temperature * gumbel_noise_like(probs)
    logging.info(f"confidence shape: {confidence.shape}")

    sorted_confidence, sorted_idx = confidence.sort(dim=-1)
    logging.info(f"sorted confidence shape: {sorted_confidence.shape}")
    logging.info(f"sorted idx shape: {sorted_idx.shape}")

    # get the cut off threshold, given the mask length
    cut_off = torch.take_along_dim(
        sorted_confidence, num_to_mask, axis=-1
    )
    logging.info(f"cut off shape: {cut_off.shape}")

    # mask out the tokens
    mask = confidence < cut_off
    logging.info(f"mask shape: {mask.shape}")

    return mask

def typical_filter(
        logits, 
        typical_mass: float = 0.95,
        typical_min_tokens: int = 1,):
    nb, nt, _ = logits.shape
    x_flat = rearrange(logits, "b t l -> (b t ) l")
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
    logits = rearrange(x_flat, "(b t) l -> b t l", t=nt)
    return logits

def sample_from_logits(
    logits,
    top_k: int = None,
    temperature: float = 1.0,
    sample: str = "multinomial",
    typical_filtering=False,
    typical_mass=0.2,
    typical_min_tokens=1,
):
    # add temperature
    logits = logits / temperature

    # add topk
    if top_k is not None and typical_filtering == False:
        v, topk_idx = logits.topk(top_k)
        logits[logits < v[..., [-1]]] = -float("inf")

    if typical_filtering:
        assert top_k is None
        nb, nt, _ = logits.shape
        x_flat = rearrange(logits, "b t l -> (b t ) l")
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
        logits = rearrange(x_flat, "(b t) l -> b t l", t=nt)

    if sample == "multinomial":
        probs = torch.softmax(logits, dim=-1)
        inferred = torch.stack([pr.multinomial(1).squeeze(-1) for pr in probs])
    elif sample == "argmax":
        inferred = torch.softmax(logits, dim=-1).argmax(dim=-1)
    elif sample == "gumbel":
        inferred = gumbel_sample(logits, dim=-1)
    else:
        raise ValueError(f"invalid sampling method: {sample}")

    return inferred


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
        z_hat = model(z_mask_latent, r)

        pred = z_hat.argmax(dim=1)
        pred = model.embedding.unflatten(pred, n_codebooks=model.n_predict_codebooks)

        print(f"model has {num_params(model)/1e6:<.3f}M parameters")
        print(f"prediction has shape {pred.shape}")
        breakpoint()

    args = argbind.parse_args()
    with argbind.scope(args):
        try_model()


