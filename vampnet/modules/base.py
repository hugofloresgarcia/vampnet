import math
from typing import Optional
from typing import Tuple
from typing import Union

import audiotools as at
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm


def log(t, eps=1e-20):
    return torch.log(t + eps)


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1.0, dim=-1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)


class VampBase(at.ml.BaseModel):
    def forward(self, x: torch.Tensor, r: torch.Tensor):
        raise NotImplementedError

    def add_noise(
        self,
        x: torch.Tensor,
        r: torch.Tensor,
        random_x: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        n_prefix: Optional[torch.Tensor] = None,
        n_suffix: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x.ndim == 3, "x must be (batch, n_codebooks, seq)"

        if mask is None:
            r = self.gamma(r)[:, None, None]
            probs = torch.ones_like(x) * r

            # if we have a prefix or suffix, set their mask prob to 0
            if n_prefix is not None:
                for i, n in enumerate(n_prefix):
                    probs[i, :, :n] = 0.0
            if n_suffix is not None:
                for i, n in enumerate(n_suffix):
                    probs[i, :, -n:] = 0.0

            mask = torch.bernoulli(probs)
            mask = mask.round().long()
            
            # if we have any conditioning codebooks, set their mask  to 0
            mask[:, : self.n_conditioning_codebooks, :] = 0
        else:
            assert mask.ndim == 3, "mask must be (batch, n_codebooks, seq)"
            assert mask.shape == x.shape, "mask must be same shape as x"

        if random_x is None:
            random_x = torch.randint_like(x, 0, self.vocab_size)

        if self.noise_mode == "mask":
            random_x = torch.full_like(x, self.mask_token)
        elif self.noise_mode == "random":
            if random_x is None:
                random_x = torch.randint_like(x, 0, self.vocab_size)
        else:
            raise ValueError(f"invalid noise mode {self.noise_mode}")

        x = x * (1 - mask) + random_x * mask
        return x, mask

    def add_truth_to_logits(
        self,
        z_true,
        z_hat,
        mask,
    ):
        if self.noise_mode == "mask":
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

    def gamma(self, r):
        return (r * torch.pi / 2).cos()

    def r_embed(self, r, max_positions=10000):
        """ """
        assert hasattr(self, "r_cond_dim"), "must set r_cond_dim before calling r_embed"

        if self.r_cond_dim > 0:
            dtype = r.dtype

            r = self.gamma(r) * max_positions
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
        if z.ndim == 2:
            z = self.embedding.unflatten(z)
        assert z.ndim == 3

        signal = at.AudioSignal(
            codec.decode(
                codec.quantizer.from_latents(self.embedding.from_codes(z, codec))[0]
            )["audio"],
            codec.sample_rate,
        )

        return signal

    @torch.no_grad()
    def sample(self, **kwargs):
        if self.noise_mode == "mask":
            return self.maskgit_sample(**kwargs)
        else:
            return self.paella_sample(**kwargs)

    def paella_sample(
        self,
        codec,
        time_steps: int = 400,
        sampling_steps: int = 12,
        start_tokens: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        device: str = "cpu",
        temperature: Union[float, Tuple[float, float]] = 0.8,
        top_k: int = None,
        sample: str = "gumbel",
        renoise_mode: str = "start",
        renoise_steps=None,
        typical_filtering=True,
        typical_mass=0.2,
        typical_min_tokens=1,
        return_signal=True,
    ):
        r = torch.linspace(0, 1, sampling_steps + 1)[:-1][:, None].to(device)
        if renoise_steps == None:
            renoise_steps = sampling_steps - 1

        if isinstance(temperature, float):
            temperature = torch.tensor(temperature).repeat(sampling_steps)
        elif isinstance(temperature, tuple):
            assert len(temperature) == 2
            l, h = temperature
            temperature = torch.linspace(l, h, sampling_steps)
        else:
            raise TypeError(f"invalid type for temperature")

        if self.n_conditioning_codebooks > 0:
            assert (
                start_tokens is not None
            ), "must provide start_tokens if n_conditioning_codebooks > 0"

        if start_tokens is None:
            if self.noise_mode == "noise":
                z = torch.randint(
                    0, self.vocab_size, size=(1, self.n_codebooks, time_steps)
                ).to(device)
            elif self.noise_mode == "mask":
                z = torch.full((1, self.n_codebooks, time_steps), self.mask_token)
        else:
            z = start_tokens
            assert (
                z.ndim == 3
            ), f"start_tokens must be shape (batch, n_codebooks, seq_len), got {z.shape}"
            assert z.shape[0] == 1, f"batch size must be 1"

        if mask is None:
            mask = torch.ones(z.shape[0], z.shape[-1]).to(device).int()

        # apply mask
        assert mask.shape == (
            z.shape[0],
            z.shape[-1],
        ), f"mask must be shape (batch, seq_len), got {mask.shape}"
        mask = mask[:, None, :]
        mask = mask.repeat(1, z.shape[1], 1)
        mask[:, : self.n_conditioning_codebooks, :] = 0.0

        if self.noise_mode == "mask":
            z_true = z.clone()

        z, mask = self.add_noise(z, r=r[0], random_x=None, mask=mask)
        z_init = z.clone()
        for i, tmpt in enumerate(temperature):
            if renoise_mode == "prev":
                z_prev = z.clone()

            latents = self.embedding.from_codes(z, codec)
            logits = self.forward(latents, r[i])

            # for mask mode
            logits = self.add_truth_to_logits(z_true, logits, mask)

            # Apply topk sampling
            logits = logits.permute(0, 2, 1)

            z = self.sample_from_logits(
                logits,
                tmpt,
                top_k,
                sample=sample,
                typical_filtering=typical_filtering,
                typical_mass=typical_mass,
                typical_min_tokens=typical_min_tokens,
            )

            # add back in conditioning codebooks
            z = self.embedding.unflatten(z, n_codebooks=self.n_predict_codebooks)
            z = torch.cat(
                [z_init[:, : self.n_conditioning_codebooks, :], z], dim=1
            ).int()

            if i < renoise_steps:
                if renoise_mode == "prev":
                    z, _ = self.add_noise(z, r[i + 1], random_x=z_prev)
                elif renoise_mode == "start":
                    z, _ = self.add_noise(z, r[i + 1], random_x=z_init)
                elif renoise_mode == "rand":
                    z, _ = self.add_noise(z, r[i + 1])
                else:
                    raise ValueError(f"Invalid renoise_mode: {renoise_mode}")

            if mask is not None:
                z = start_tokens * (1 - mask) + z * mask

        if return_signal:
            return self.to_signal(z, codec)
        else:
            return z

    def maskgit_sample(
        self,
        codec,
        time_steps: int = 300,
        sampling_steps: int = 24,
        start_tokens: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        temperature: Union[float, Tuple[float, float]] = 0.8,
        top_k: int = None,
        sample: str = "multinomial",
        typical_filtering=False,
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

        def flatten(codes):
            return rearrange(codes, "b c t -> b (t c)")

        def unflatten(codes, c):
            return rearrange(codes, "b (t c) -> b c t", c=c)

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
        keep_mask = flatten(keep_mask)

        # our r steps
        r_steps = torch.linspace(0, 1, sampling_steps + 1)[1:].to(self.device)

        # how many tokens did we keep on init?
        num_kept_on_init = keep_mask.sum()

        # how many codebooks are we inferring vs conditioning on?
        n_infer_codebooks = self.n_codebooks - self.n_conditioning_codebooks

        for i in tqdm(range(sampling_steps)):
            # our current temperature
            tmpt = temperature[i]

            # our current schedule step
            r = r_steps[i : i + 1]

            with torch.inference_mode():
                # mask our z
                keep_mask_unflat = unflatten(keep_mask, c=self.n_codebooks)
                z_masked = z.masked_fill(~keep_mask_unflat.bool(), self.mask_token)

                # get latents
                latents = self.embedding.from_codes(z_masked, codec)

                # infer from latents
                logits = self.forward(latents, r)
                logits = logits.permute(0, 2, 1)  # b, seq, prob

                # the schedule determines how many samples to keep
                num_tokens_to_infer = (z.shape[-1] * z.shape[-2]) - num_kept_on_init
                num_to_keep = num_kept_on_init + int(
                    num_tokens_to_infer * (self.gamma(1 - r))
                )

                # figure out which logits we wanna keep
                if num_to_keep > 0:
                    probs = logits.softmax(dim=-1)

                    keep_probs = F.one_hot(z, self.vocab_size)[:, :, :]

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

                z_inferred = self.sample_from_logits(
                    logits=logits,
                    top_k=top_k,
                    temperature=tmpt,
                    sample=sample,
                    typical_filtering=typical_filtering,
                    typical_mass=typical_mass,
                    typical_min_tokens=typical_min_tokens,
                )

                z = rearrange(z_inferred, "b (t c) -> b c t", c=self.n_codebooks)

                # add conditioning codebooks back
                # z = torch.cat([z[:, :self.n_conditioning_codebooks, :], z_inferred], dim=1)

        if return_signal:
            return self.to_signal(z, codec)
        else:
            return z

    def sample_from_logits(
        self,
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
        if top_k is not None:
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
            inferred = torch.softmax(probs, dim=-1).argmax(dim=-1)
        elif sample == "gumbel":
            inferred = gumbel_sample(logits, dim=-1)
        else:
            raise ValueError(f"invalid sampling method: {sample}")

        return inferred
