import logging
import math
import typing as tp

import demucs
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from einops import rearrange
from torch import Tensor

logger = logging.getLogger()


def length_to_mask(
    lengths: torch.Tensor, max_len: tp.Optional[int] = None
) -> torch.Tensor:
    """Utility function to convert a tensor of sequence lengths to a mask (useful when working on padded sequences).
    For example: [3, 5] => [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]

    Args:
        lengths (torch.Tensor): tensor with lengths
        max_len (int): can set the max length manually. Defaults to None.
    Returns:
        torch.Tensor: mask with 0s where there is pad tokens else 1s
    """
    assert len(lengths.shape) == 1, "Length shape should be 1 dimensional."
    final_length = lengths.max().item() if not max_len else max_len
    final_length = max(
        final_length, 1
    )  # if all seqs are of len zero we don't want a zero-size tensor
    return torch.arange(final_length)[None, :].to(lengths.device) < lengths[:, None]


class WavCondition(tp.NamedTuple):
    wav: Tensor
    length: Tensor
    path: tp.List[tp.Optional[str]] = []


ConditionType = tp.Tuple[Tensor, Tensor]  # condition, mask


def nullify_condition(condition: ConditionType, dim: int = 1):
    """This function transforms an input condition to a null condition.
    The way it is done by converting it to a single zero vector similarly
    to how it is done inside WhiteSpaceTokenizer and NoopTokenizer.

    Args:
        condition (ConditionType): a tuple of condition and mask (tp.Tuple[Tensor, Tensor])
        dim (int): the dimension that will be truncated (should be the time dimension)
        WARNING!: dim should not be the batch dimension!
    Returns:
        ConditionType: a tuple of null condition and mask
    """
    assert dim != 0, "dim cannot be the batch dimension!"
    assert (
        type(condition) == tuple
        and type(condition[0]) == Tensor
        and type(condition[1]) == Tensor
    ), "'nullify_condition' got an unexpected input type!"
    cond, mask = condition
    B = cond.shape[0]
    last_dim = cond.dim() - 1
    out = cond.transpose(dim, last_dim)
    out = 0.0 * out[..., :1]
    out = out.transpose(dim, last_dim)
    mask = torch.zeros((B, 1), device=out.device).int()
    assert cond.dim() == out.dim()
    return out, mask


def nullify_wav(wav: Tensor) -> WavCondition:
    """Create a nullified WavCondition from a wav tensor with appropriate shape.

    Args:
        wav (Tensor): tensor of shape [B, T]
    Returns:
        WavCondition: wav condition with nullified wav.
    """
    null_wav, _ = nullify_condition((wav, torch.zeros_like(wav)), dim=wav.dim() - 1)
    return WavCondition(
        wav=null_wav,
        length=torch.tensor([0] * wav.shape[0], device=wav.device),
        path=["null_wav"] * wav.shape[0],
    )


class BaseConditioner(nn.Module):
    """Base model for all conditioner modules. We allow the output dim to be different
    than the hidden dim for two reasons: 1) keep our LUTs small when the vocab is large;
    2) make all condition dims consistent.

    Args:
        dim (int): Hidden dim of the model (text-encoder/LUT).
        output_dim (int): Output dim of the conditioner.
    """

    def __init__(self, dim, output_dim):
        super().__init__()
        self.dim = dim
        self.output_dim = output_dim
        self.output_proj = nn.Linear(dim, output_dim)

    def tokenize(self, *args, **kwargs) -> tp.Any:
        """Should be any part of the processing that will lead to a synchronization
        point, e.g. BPE tokenization with transfer to the GPU.

        The returned value will be saved and return later when calling forward().
        """
        raise NotImplementedError()

    def forward(self, inputs: tp.Any) -> ConditionType:
        """Gets input that should be used as conditioning (e.g, genre, description or a waveform).
        Outputs a ConditionType, after the input data was embedded as a dense vector.

        Returns:
            ConditionType:
                - A tensor of size [B, T, D] where B is the batch size, T is the length of the
                  output embedding and D is the dimension of the embedding.
                - And a mask indicating where the padding tokens.
        """
        raise NotImplementedError()


class WaveformConditioner(BaseConditioner):
    """Base class for all conditioners that take a waveform as input.
    Classes that inherit must implement `_get_wav_embedding` that outputs
    a continuous tensor, and `_downsampling_factor` that returns the down-sampling
    factor of the embedding model.

    Args:
        dim (int): The internal representation dimension.
        output_dim (int): Output dimension.
        device (tp.Union[torch.device, str]): Device.
    """

    def __init__(self, dim: int, output_dim: int, device: tp.Union[torch.device, str]):
        super().__init__(dim, output_dim)
        self.device = device
        self.to(device)

    def tokenize(self, wav_length: WavCondition) -> WavCondition:
        wav, length, path = wav_length
        assert length is not None
        return WavCondition(wav.to(self.device), length.to(self.device), path)

    def _get_wav_embedding(self, wav: Tensor) -> Tensor:
        """Gets as input a wav and returns a dense vector of conditions."""
        raise NotImplementedError()

    def _downsampling_factor(self):
        """Returns the downsampling factor of the embedding model."""
        raise NotImplementedError()

    def forward(self, inputs: WavCondition) -> ConditionType:
        """
        Args:
            input (WavCondition): Tuple of (waveform, lengths).
        Returns:
            ConditionType: Dense vector representing the conditioning along with its' mask.
        """
        wav, lengths, path = inputs
        with torch.no_grad():
            embeds = self._get_wav_embedding(wav)

        embeds = embeds.to(self.output_proj.weight)
        embeds = self.output_proj(embeds)

        assert lengths is None
        # if lengths is not None:
        #     lengths = lengths / self._downsampling_factor()
        #     mask = length_to_mask(lengths, max_len=embeds.shape[1]).int()  # type: ignore
        # else:
        #     mask = torch.ones_like(embeds)
        # embeds = (embeds * mask.unsqueeze(2).to(self.device))

        return embeds, torch.ones_like(embeds)


class ChromaStemConditioner(WaveformConditioner):
    """Chroma conditioner that uses DEMUCS to first filter out drums and bass. The is followed by
    the insight the drums and bass often dominate the chroma, leading to the chroma not containing the
    information about melody.

    Args:
        output_dim (int): Output dimension for the conditioner.
        sample_rate (int): Sample rate for the chroma extractor.
        n_chroma (int): Number of chroma for the chroma extractor.
        radix2_exp (int): Radix2 exponent for the chroma extractor.
        duration (float): Duration used during training. This is later used for correct padding
            in case we are using chroma as prefix.
        match_len_on_eval (bool, optional): If True then all chromas are padded to the training
            duration. Defaults to False.
        eval_wavs (str, optional): Path to a json egg with waveform, this waveforms are used as
            conditions during eval (for cases where we don't want to leak test conditions like MusicCaps).
            Defaults to None.
        n_eval_wavs (int, optional): Limits the number of waveforms used for conditioning. Defaults to 0.
        device (tp.Union[torch.device, str], optional): Device for the conditioner.
        **kwargs: Additional parameters for the chroma extractor.
    """

    def __init__(
        self,
        output_dim: int,
        sample_rate: int,
        n_chroma: int,
        duration: float,
        radix2_exp: int = 12,
        match_len_on_eval: bool = True,
        eval_wavs: tp.Optional[str] = None,
        n_eval_wavs: int = 0,
        device: tp.Union[torch.device, str] = "cpu",
        **kwargs,
    ):
        from demucs import pretrained

        self.__dict__["demucs"] = pretrained.get_model("htdemucs").to(device)

        super().__init__(dim=n_chroma, output_dim=output_dim, device=device)

        self.sample_rate = sample_rate
        self.match_len_on_eval = match_len_on_eval
        self.duration = duration
        self.stem2idx = {"drums": 0, "bass": 1, "other": 2, "vocal": 3}
        self.stem_idx = torch.LongTensor(
            [self.stem2idx["vocal"], self.stem2idx["other"]]
        ).to(device)
        self.chroma = ChromaExtractor(
            sample_rate=sample_rate,
            n_chroma=n_chroma,
            radix2_exp=radix2_exp,
            device=device,
            **kwargs,
        )
        self.chroma_len = self._get_chroma_len()

    def _downsampling_factor(self):
        return self.chroma.winhop

    def _get_chroma_len(self):
        """Get length of chroma during training"""
        dummy_wav = torch.zeros(
            1, int(self.sample_rate * self.duration), device=self.device
        )
        dummy_chr = self.chroma(dummy_wav)
        return dummy_chr.shape[1]

    @torch.no_grad()
    def _get_filtered_wav(self, wav):
        from demucs.apply import apply_model
        from demucs.audio import convert_audio

        wav = convert_audio(
            wav, self.sample_rate, self.demucs.samplerate, self.demucs.audio_channels
        )
        stems = apply_model(self.demucs, wav, device=self.device)
        stems = stems[:, self.stem_idx]  # extract stem
        stems = stems.sum(1)  # merge extracted stems
        stems = stems.mean(1, keepdim=True)  # mono
        stems = convert_audio(stems, self.demucs.samplerate, self.sample_rate, 1)
        return stems

    @torch.no_grad()
    def _get_wav_embedding(self, wav):
        # avoid 0-size tensors when we are working with null conds
        if wav.shape[-1] == 1:
            return self.chroma(wav)

        stems = self._get_filtered_wav(wav)
        chroma = self.chroma(stems)

        if self.match_len_on_eval:
            b, t, c = chroma.shape
            if t > self.chroma_len:
                chroma = chroma[:, : self.chroma_len]
                print(f"chroma was truncated! ({t} -> {chroma.shape[1]})")
            elif t < self.chroma_len:
                # chroma = F.pad(chroma, (0, 0, 0, self.chroma_len - t))
                n_repeat = int(math.ceil(self.chroma_len / t))
                chroma = chroma.repeat(1, n_repeat, 1)
                chroma = chroma[:, : self.chroma_len]
                print(f"chroma was zero-padded! ({t} -> {chroma.shape[1]})")

        return chroma


class ChromaExtractor(nn.Module):
    """Chroma extraction class, handles chroma extraction and quantization.

    Args:
        sample_rate (int): Sample rate.
        n_chroma (int): Number of chroma to consider.
        radix2_exp (int): Radix2 exponent.
        nfft (tp.Optional[int], optional): Number of FFT.
        winlen (tp.Optional[int], optional): Window length.
        winhop (tp.Optional[int], optional): Window hop size.
        argmax (bool, optional): Whether to use argmax. Defaults to False.
        norm (float, optional): Norm for chroma normalization. Defaults to inf.
        device (tp.Union[torch.device, str], optional): Device to use. Defaults to cpu.
    """

    def __init__(
        self,
        sample_rate: int,
        n_chroma: int = 12,
        radix2_exp: int = 12,
        nfft: tp.Optional[int] = None,
        winlen: tp.Optional[int] = None,
        winhop: tp.Optional[int] = None,
        argmax: bool = False,
        norm: float = torch.inf,
        device: tp.Union[torch.device, str] = "cpu",
    ):
        super().__init__()
        from librosa import filters

        self.device = device
        self.winlen = winlen or 2**radix2_exp
        self.nfft = nfft or self.winlen
        self.winhop = winhop or (self.winlen // 4)
        self.sr = sample_rate
        self.n_chroma = n_chroma
        self.norm = norm
        self.argmax = argmax

        self.window = torch.hann_window(self.winlen).to(device)
        self.fbanks = torch.from_numpy(
            filters.chroma(
                sr=sample_rate, n_fft=self.nfft, tuning=0, n_chroma=self.n_chroma
            )
        ).to(device)
        self.spec = torchaudio.transforms.Spectrogram(
            n_fft=self.nfft,
            win_length=self.winlen,
            hop_length=self.winhop,
            power=2,
            center=True,
            pad=0,
            normalized=True,
        ).to(device)

    def forward(self, wav):
        T = wav.shape[-1]
        # in case we are getting a wav that was dropped out (nullified)
        # make sure wav length is no less that nfft
        if T < self.nfft:
            pad = self.nfft - T
            r = 0 if pad % 2 == 0 else 1
            wav = F.pad(wav, (pad // 2, pad // 2 + r), "constant", 0)
            assert (
                wav.shape[-1] == self.nfft
            ), f"expected len {self.nfft} but got {wav.shape[-1]}"
        spec = self.spec(wav).squeeze(1)
        raw_chroma = torch.einsum("cf,...ft->...ct", self.fbanks, spec)
        norm_chroma = torch.nn.functional.normalize(
            raw_chroma, p=self.norm, dim=-2, eps=1e-6
        )
        norm_chroma = rearrange(norm_chroma, "b d t -> b t d")

        if self.argmax:
            idx = norm_chroma.argmax(-1, keepdims=True)
            norm_chroma[:] = 0
            norm_chroma.scatter_(dim=-1, index=idx, value=1)

        return norm_chroma


if __name__ == "__main__":
    from audiotools import AudioSignal

    sig = (
        AudioSignal.salient_excerpt(
            "/media/CHONK/hugo/loras/dariacore/c0ncernn - 1235.mp3", duration=10
        )
        .to("cuda:0")
        .to_mono()
    )
    hop = 768

    conditioner = ChromaStemConditioner(
        output_dim=512,
        sample_rate=sig.sample_rate,
        n_chroma=12,
        duration=10.0,
        device=sig.device,
        winhop=hop,
    )

    cond, cond_mask = conditioner((sig.samples, None, None))

    print(cond.shape)
    print(cond)
