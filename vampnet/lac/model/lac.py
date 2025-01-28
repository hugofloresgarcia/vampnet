import math
from typing import List
from typing import Union

import numpy as np
import torch
from audiotools import AudioSignal
from audiotools.ml import BaseModel
from torch import nn
from torch.nn.utils import weight_norm

from .base import CodecMixin
from vampnet.lac.nn.quantize import ResidualVectorQuantize
from vampnet.lac.nn.layers import WNConv1d
from vampnet.lac.nn.layers import WNConvTranspose1d
from vampnet.lac.nn.layers import Snake1d


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


class EncoderLayer(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        return x + self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            EncoderLayer(dim // 2, dilation=1),
            EncoderLayer(dim // 2, dilation=3),
            EncoderLayer(dim // 2, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=stride // 2,
            ),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
    ):
        super().__init__()
        # Create first convolution
        self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for stride in strides:
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride)]

        # Create last convolution
        self.block += [
            Snake1d(d_model),
            WNConv1d(d_model, d_model, kernel_size=3, padding=1),
        ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x):
        return self.block(x)


class ResidualLayer(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.layer = nn.Sequential(
            Snake1d(channels),
            WNConv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=((kernel_size - 1) * dilation) // 2,
            ),
            Snake1d(channels),
            WNConv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                dilation=1,
                padding=(kernel_size - 1) // 2,
            ),
        )

    def forward(self, x):
        return x + self.layer(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilations):
        super().__init__()
        self.block = nn.Sequential(
            *[ResidualLayer(channels, kernel_size, dilation) for dilation in dilations]
        )

    def forward(self, x):
        return self.block(x)


class MRFBlock(nn.Module):
    def __init__(self, channels, kernel_sizes, dilations):
        super().__init__()
        self.mrf_blocks = nn.ModuleList(
            [
                ResidualBlock(channels, kernel_size, dilation)
                for kernel_size, dilation in zip(kernel_sizes, dilations)
            ]
        )

    def forward(self, x):
        x_sum = 0
        for layer in self.mrf_blocks:
            x_sum += layer(x)
        return x_sum / len(self.mrf_blocks)


class Block(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        stride: int = 1,
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    ):
        super().__init__()

        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=stride * 2,
                stride=stride,
                padding=stride // 2,
            ),
            MRFBlock(
                output_dim,
                resblock_kernel_sizes,
                resblock_dilation_sizes,
            ),
        )
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        rates,
        d_out: int = 1,
    ):
        super().__init__()

        # Add first conv layer
        layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]

        # Add upsampling + MRF blocks
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [Block(input_dim, output_dim, stride)]

        # Add final conv layer
        layers += [
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class LAC(BaseModel, CodecMixin):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 4, 8, 8],
        decoder_dim: int = 512,
        decoder_rates: List[int] = [4, 4, 4, 2, 2, 2],
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: bool = False,
        sample_rate: int = 44100,
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate

        self.hop_length = np.prod(decoder_rates)
        self.encoder = Encoder(encoder_dim, encoder_rates)

        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.quantizer = ResidualVectorQuantize(
            self.encoder.enc_dim, 
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=quantizer_dropout,
        )

        self.decoder = Decoder(
            self.encoder.enc_dim,
            decoder_dim,
            decoder_rates,
        )
        self.sample_rate = sample_rate
        self.apply(init_weights)

    def preprocess(self, audio_data, sample_rate):
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate

        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))
        return audio_data, length

    def encode(
        self, 
        audio_data: torch.Tensor, 
        sample_rate: int = None,
        n_quantizers: int = None,
    ):
        out = {}
        audio_data, length = self.preprocess(audio_data, sample_rate)
        out["length"] = length
        
        out["z"] = self.encoder(audio_data)
        out.update(self.quantizer(out["z"], n_quantizers))
        return out

    def decode(
        self, 
        z: torch.Tensor,
        length: int = None
    ):
        out = {}
        x = self.decoder(z)
        out["audio"] = x[..., :length]
        return out

    def forward(
        self,
        audio_data: torch.Tensor,
        sample_rate: int = None,
        n_quantizers: int = None,
    ):
        out = {}
        out.update(self.encode(audio_data, sample_rate, n_quantizers))
        out.update(self.decode(out["z"], out["length"]))
        return out


if __name__ == "__main__":
    import numpy as np
    from functools import partial

    x = torch.randn(1, 1, 30 * 44100).float()
    x = AudioSignal(x, 44100)

    print(x)
    model = LAC().cuda()

    for n, m in model.named_modules():
        o = m.extra_repr()
        p = sum([np.prod(p.size()) for p in m.parameters()])
        fn = lambda o, p: o + f" {p/1e6:<.3f}M params."
        setattr(m, "extra_repr", partial(fn, o=o, p=p))
    print(model)
    print("Total # of params: ", sum([np.prod(p.size()) for p in model.parameters()]))

    output = model.encode(x, verbose=True)

    length = 88200 * 2
    x = torch.randn(1, 1, length).to(model.device)
    x.requires_grad_(True)
    x.retain_grad()

    # Make a forward pass
    out = model(x)["audio"]

    # Create gradient variable
    grad = torch.zeros_like(out)
    grad[:, :, grad.shape[-1] // 2] = 1

    # Make a backward pass
    out.backward(grad)

    # Check non-zero values
    gradmap = x.grad.squeeze(0)
    gradmap = (gradmap != 0).sum(0)  # sum across features
    rf = (gradmap != 0).sum()

    print(f"Receptive field: {rf.item()}")
