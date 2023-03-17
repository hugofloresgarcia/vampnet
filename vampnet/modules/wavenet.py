import torch.nn as nn
from einops import rearrange

from voicegpt.nn import WaveNet

class AutoregMLP(nn.Module):
    """Implements an autoregressive ConvNet decoder
    Refer to SampleRNN (https://arxiv.org/abs/1612.07837) for motivation
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_fine_tokens: int = 6,
        n_tokens: int = 9,
        dropout: float = 0.1,
        activation: str = "gelu",
        causal: bool = True,
    ):
        super().__init__()
        self.n_fine = n_fine_tokens
        self.n_layers = n_layers
        self.upsampler = nn.Linear(d_model, d_model * n_fine_tokens)

        self.wavenet = WaveNet(
            d_model,
            d_model,
            d_model,
            n_layers,
            n_fine_tokens,
            dropout=dropout,
            activation=activation,
            causal=causal,
        )
        self.ff_output = nn.Linear(d_model, vocab_size * n_tokens, bias=False)

    def time_upsample(self, h_t_coarse):
        """Upsamples the conditioning hidden states to match the time resolution
        of output tokens
        Parameters
        ----------
        h_t_coarse : Tensor[B x T_coarse x D]
            Conditioning hidden states in coarse time-scale
        Returns
        -------
        Tensor[B x T_fine x D]
            Conditioning hidden states in fine time-scale
        """
        # Upsample the transformer hidden states to fine scale
        h_t_fine = rearrange(
            self.upsampler(h_t_coarse), "b t (n d) -> b (t n) d", n=self.n_fine
        )
        return h_t_fine

    def decode_logits(self, x_tm1, h_t_fine):
        """Decodes output logits conditioned on previous output
        tokens (upto timestep t-1) and conditioning hidden states
        using an autoregressive WaveNet
        Parameters
        ----------
        x_tm1 : Tensor[B x T x D]
        h_t_fine : Tensor[B x T x D]
        Returns
        -------
        Tensor[B x T x vocab_size]
            Predicted logits
        """

        # Compute wavenet layers and predict logits
        o_t = self.wavenet(x_tm1, h_t_fine)
        return self.ff_output(o_t)

    def forward(self, x_tm1, h_t_coarse):
        """Computes autoregressive conditional probability distribution
        using a WaveNet decoder
        Parameters
        ----------
        x_tm1 : Tensor[B x T_fine x D]
            Embeddings of tokens at fine time-scale
        h_t_coarse : Tensor[B x T_coarse x D]
            Hidden states at coarse time scale
        Returns
        -------
        Tensor[B x T_fine x vocab_size]
            Predicted logits at fine time-scale
        """
        h_t_fine = self.time_upsample(h_t_coarse)
        return self.decode_logits(x_tm1, h_t_fine)