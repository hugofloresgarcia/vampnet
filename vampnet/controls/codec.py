
from pathlib import Path
import torch
import numpy as np
import tqdm

from audiotools import AudioSignal
from dac.utils import load_model 
from dac.model.dac import DAC
import vampnet


def receptive_field(model):
    """
    Computes the size, stride and padding of the given model's receptive
    field under the assumption that all its Conv1d and TransposeConv1d
    layers are applied in sequence.
    """
    total_size, total_stride, total_padding = 1, 1, 0
    for layer in model.modules():
        if isinstance(layer, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
            layer_size = layer.dilation[0] * (layer.kernel_size[0] - 1) + 1
        if isinstance(layer, torch.nn.Conv1d):
            # update size
            total_size += (layer_size - 1) * total_stride
            # update padding
            total_padding += layer.padding[0] * total_stride
            # update stride
            total_stride *= layer.stride[0]
        elif isinstance(layer, torch.nn.ConvTranspose1d):
            # update stride
            total_stride /= layer.stride[0]
            # update padding
            total_padding += (layer_size - layer.padding[0]) * total_stride
            # update size
            total_size += (layer_size - 1) * total_stride
    return total_size, total_stride, total_padding


@torch.inference_mode()
def compress(model, device, audio, win_duration, n_quantizers=None):
    """Encodes the given audio signal, returns the codes."""
    # right-pad to the next multiple of hop length
    # (as the model's internal padding is short by one hop length)
    remainder = audio.shape[-1] % model.hop_length
    right_pad = model.hop_length - remainder if remainder else 0
    model.to(device)
    if not win_duration:
        model.padding = True
        if right_pad:
            audio.zero_pad(0, right_pad)
        samples = audio.audio_data.to(device)
        codes = model.encode(samples, n_quantizers)["codes"]
        codes = codes.permute(2, 1, 0).short()  # -> time, quantizers, channels
    else:
        # determine receptive field of encoder
        model.padding = True
        field_size, stride, padding = receptive_field(model.encoder)
        model.padding = False
        # determine the window size to use
        # - the maximum samples the user wants to read at once
        win_size = int(win_duration * model.sample_rate)
        # - how many code frames we would get from this
        num_codes = (win_size - field_size + stride) // stride
        # - how many samples are actually involved in that
        win_size = field_size + (num_codes - 1) * stride
        # determine the hop size to use
        hop_size = num_codes * stride
        # finally process the input
        codes = []
        audio_size = audio.audio_data.size(-1)
        for start_position in tqdm.trange(-padding,
                                          audio_size + padding + right_pad,
                                          hop_size,
                                          leave=False):
            # extract chunk
            chunk = audio[..., max(0, start_position):start_position + win_size]
            # zero-pad the first chunk(s)
            if start_position < 0:
                chunk.zero_pad(-start_position, 0)
            chunk_size = chunk.audio_data.size(-1)
            # skip the last chunk if it would not have yielded any output
            if chunk_size + padding + right_pad < field_size:
                continue
            # pad the last chunk(s) to the full window size if needed
            if chunk_size < win_size:
                chunk.zero_pad(0, win_size - chunk_size)
            # process chunk
            samples = chunk.audio_data.to(device)
            c = model.encode(samples, n_quantizers)["codes"].cpu()
            c = c.permute(2, 1, 0)  # -> time, quantizers, channels
            # remove excess frames from padding if needed
            if chunk_size + padding + right_pad < win_size:
                chunk_codes = (chunk_size + padding + right_pad - field_size + stride) // stride
                c = c[:chunk_codes]
            codes.append(c.short())
        codes = torch.cat(codes, dim=0)
    return codes.contiguous()

def _load_from_hub():
    from huggingface_hub import hf_hub_download
    # repo_id, model_name = model_id.split(":")
    # download the model
    filename = "codec.pth"
    # filename = vampnet.MODEL_FILE.name
    # subfolder = vampnet.MODEL_FILE.parent.relative_to(vampnet.ROOT)
    model_path = hf_hub_download(
        "hugggof/codec",
        filename=filename,
        # subfolder=subfolder,
        cache_dir=vampnet.MODELS_DIR / "hub_cache")
    # load the model
    return load_model(load_path=str(model_path))

def _load_codec():
    codec = _load_from_hub()
    assert codec.sample_rate == vampnet.SAMPLE_RATE
    assert codec.hop_length == vampnet.HOP_SIZE
    codec.to(vampnet.DEVICE)
    return codec


def load_codec():
    if not hasattr(vampnet.controls.codec, '_codec'):
        vampnet.controls.codec._codec = _load_codec()
    return vampnet.controls.codec._codec

from dataclasses import dataclass
@dataclass
class DACControl(vampnet.controls.Control):
    ctrl: torch.Tensor
    name = 'dac'
    hop_size: int = vampnet.HOP_SIZE
    ext: str = ".dac"
    metadata: dict = None

    @classmethod
    def from_signal(cls, sig: AudioSignal, device=vampnet.DEVICE):

        from dac.model.base import DACFile
        metadata = {}
        metadata["original_length"] = int(sig.samples.shape[-1])
        metadata["input_db"] = float(sig.ffmpeg_loudness())
        metadata["win_duration"] = vampnet.HOP_SIZE * 10000 / vampnet.SAMPLE_RATE

        sig = sig.normalize(vampnet.LOUD_NORM).ensure_max_of_audio()
        sig.samples = sig.samples.view(-1, 1, sig.samples.shape[-1])

        codes = compress(
                load_codec(), 
                device, 
                sig,
                win_duration=metadata["win_duration"], # like 30s 
        ) # (nt, nch, nc
        codes = codes.permute(2, 1, 0) # (nch, nc, nt)

        return cls(codes, metadata=metadata)