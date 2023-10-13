import typing as tp
from typing import Tuple, Dict, Optional, List
import csv

import torchaudio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import numpy as np

from audiotools import AudioSignal

from einops import rearrange

def resample_embeddings(
        embs: torch.Tensor, n_t: int,
        method: str = 'nearest'
    ) -> torch.Tensor:
    """
    Resample embeddings in time to meet a given target number of timesteps.
    Assumes that the embeddings are uniformly sampled in time.

    Parameters:
    - embs: A tensor of shape (batch, dim, time) containing the embeddings.
    - n_t: the target time length.
    - method: Interpolation method, either 'linear' or 'nearest'.

    Returns:
    - A tensor of shape (batch, dim, new_time) containing the resampled embeddings.
    """
    assert method in ['linear', 'nearest'], "Method must be 'linear' or 'nearest'."

    B, D, T = embs.shape    
    if method == 'linear':
        # Using F.interpolate for linear interpolation
        resampled_embs = F.interpolate(
            embs.unsqueeze(0), 
            size=n_t, 
            mode='linear', 
            align_corners=True
        ).squeeze(0)
        
    elif method == 'nearest':
        new_time_indices = torch.linspace(0, T - 1, steps=n_t, device=embs.device)
        nearest_indices = torch.round(new_time_indices).long()
        resampled_embs = embs[:, :, nearest_indices]

    return resampled_embs


class WaveformConditioner:

    @property
    def keys(self) -> List[str]:
        return NotImplementedError()

    def condition(self, sig: AudioSignal) -> Dict[str, Tensor]:
        """Gets as input a wav and returns a dense vector of conditions."""
        raise NotImplementedError()


    @property
    def dim(self):
        """Returns the dimensionality of the embedding."""
        raise NotImplementedError()


class ChromaExtractor(nn.Module):

    def __init__(self, sample_rate: int, n_chroma: int = 12, radix2_exp: int = 12,
                 nfft: tp.Optional[int] = None, winlen: tp.Optional[int] = None, winhop: tp.Optional[int] = None,
                 argmax: bool = False, norm: float = torch.inf, device: tp.Union[torch.device, str] = "cpu"):
        super().__init__()
        from librosa import filters

        self.device = device
        self.winlen = winlen or 2 ** radix2_exp
        self.nfft = nfft or self.winlen
        self.winhop = winhop or (self.winlen // 4)
        self.sr = sample_rate
        self.n_chroma = n_chroma
        self.norm = norm
        self.argmax = argmax

        self.window = torch.hann_window(self.winlen).to(device)
        self.fbanks = torch.from_numpy(filters.chroma(sr=sample_rate, n_fft=self.nfft, tuning=0,
                                                      n_chroma=self.n_chroma)).to(device)
        self.spec = torchaudio.transforms.Spectrogram(n_fft=self.nfft, win_length=self.winlen,
                                                      hop_length=self.winhop, power=2, center=True,
                                                      pad=0, normalized=True).to(device)

    def forward(self, wav):
        T = wav.shape[-1]
        # in case we are getting a wav that was dropped out (nullified)
        # make sure wav length is no less that nfft
        if T < self.nfft:
            pad = self.nfft - T
            r = 0 if pad % 2 == 0 else 1
            wav = F.pad(wav, (pad // 2, pad // 2 + r), 'constant', 0)
            assert wav.shape[-1] == self.nfft, f'expected len {self.nfft} but got {wav.shape[-1]}'
        spec = self.spec(wav).squeeze(1)
        raw_chroma = torch.einsum("cf,...ft->...ct", self.fbanks, spec)
        norm_chroma = torch.nn.functional.normalize(raw_chroma, p=self.norm, dim=-2, eps=1e-6)
        norm_chroma = rearrange(norm_chroma, "b d t -> b t d")

        if self.argmax:
            idx = norm_chroma.argmax(-1, keepdims=True)
            norm_chroma[:] = 0
            norm_chroma.scatter_(dim=-1, index=idx, value=1)

        return norm_chroma


class ChromaStemConditioner(WaveformConditioner):

    def __init__(self, 
            sample_rate: int = 44100, 
            n_chroma: int = 36, 
            radix2_exp: int=12,
            hop: int=512,
            device: str = "cuda",
            **kwargs
        ):
        from demucs import pretrained
        self.__dict__["demucs"] = pretrained.get_model('htdemucs').to(device)

        self.sample_rate = sample_rate
        self.stem2idx = {'drums': 0, 'bass': 1, 'other': 2, 'vocal': 3}
        self.stem_idx = torch.LongTensor([self.stem2idx['vocal'], self.stem2idx['other']])
        self.chroma = ChromaExtractor(
            sample_rate=sample_rate, n_chroma=n_chroma, 
            radix2_exp=radix2_exp, winhop=hop,
            **kwargs
        ).to(device)

        self.device = device
    
    def to(self, device):
        self.chroma.to(device)
        self.demucs.to(device)
        self.device = device
        return self

    @property
    def dim(self):
        return self.chroma.n_chroma


    @property
    def hop(self):
        return self.chroma.winhop


    @torch.inference_mode()
    def _get_filtered_wav(self, wav):
        from demucs.apply import apply_model
        from demucs.audio import convert_audio

        wav = convert_audio(wav, self.sample_rate, self.demucs.samplerate, self.demucs.audio_channels)
        stems = apply_model(self.demucs, wav)
        stems = stems[:, self.stem_idx]  # extract stem
        stems = stems.sum(1)  # merge extracted stems
        stems = stems.mean(1, keepdim=True)  # mono
        stems = convert_audio(stems, self.demucs.samplerate, self.sample_rate, 1)
        return stems


    @torch.no_grad()
    def condition(self, sig: AudioSignal):
        wav = sig.samples.to(self.device)
        # avoid 0-size tensors when we are working with null conds
        if wav.shape[-1] == 1:
            return self.chroma(wav)

        stems = self._get_filtered_wav(wav)
        chroma = self.chroma(stems)

        return {"chroma": chroma}

    @property
    def keys(self,):
        return ["chroma"]
    

# Find the name of the class with the top score when mean-aggregated across frames.
def class_names_from_csv(class_map_csv_text):
  """Returns list of class names corresponding to score vector."""
  class_names = []
  with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      class_names.append(row['display_name'])

  return class_names


class YamnetConditioner(WaveformConditioner):
    
    def __init__(self,
        confidence_threshold: Optional[float] = None, 
    ):
        import tensorflow as tf
        import tensorflow_hub as hub
        
        # Load the model.
        self.model = hub.load('https://tfhub.dev/google/yamnet/1')

        class_map_path = self.model.class_map_path().numpy()

        self._class_names = class_names_from_csv(class_map_path)
        self.yamnet_sample_rate = 16000
        self.chunk_size = int(self.yamnet_sample_rate * 0.975)

        self.confidence_threshold = confidence_threshold

    @property
    def class_names(self):
        return self._class_names
    
    @property
    def build_cond(self, class_probs: Dict[str, float]):
        cond = torch.zeros(len(self.class_names))
        for class_name, prob in class_probs:
            cond[self.class_names.index(class_name)] = prob
        return cond

    @torch.inference_mode()
    def condition(self, sig: AudioSignal):
        # have to resample to 16k
        sig = sig.resample(self.yamnet_sample_rate).to_mono()
        audio = sig.samples

        # Run the model, check the output.
        scores, embeddings, spectrogram = self.model(audio[0][0])

        scores = scores.numpy()
        embeddings = embeddings.numpy()
        # spectrogram = spectrogram.numpy()
        # infered_class = self.class_names[scores.mean(axis=0).argmax()]

        scores = torch.from_numpy(scores).float()
        embeddings = torch.from_numpy(embeddings).float()

        return {"scores": scores, "embeddings": embeddings}

    @property
    def keys(self,):
        return ["scores", "embeddings"]


class MFCCConditioner(WaveformConditioner):
    
    pass


class ConditionEmbedder(nn.Module):

    def __init__(self, 
        input_dim: int,
        output_dim: int,
        resample_method: str = 'nearest',
    ):
        super().__init__()
        self.resample_method = resample_method
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x: Tensor, target_length: int) -> Tensor:
        x = resample_embeddings(x, target_length, method=self.resample_method)
        x =  self.proj(x)
        return x
    

import zipfile
import json
from dataclasses import dataclass, fields
@dataclass
class ConditionFeatures:
    audio_path: str
    features: Dict[str, np.array]
    metadata: dict

    def save(self, path):
        """Save the Embedding object to a given path as a zip file."""
        with zipfile.ZipFile(path, 'w') as archive:
            
            # Save numpy array
            for key, array in self.features.items():
                with archive.open(f'{key}.npy', 'w') as f:
                    np.save(f, array)

            # Save non-numpy data as json
            non_numpy_data = {f.name: getattr(self, f.name) for f in fields(self) if f.name != 'features'}
            non_numpy_data["_keys"] = list(self.features.keys())
            with archive.open('data.json', 'w') as f:
                f.write(json.dumps(non_numpy_data).encode('utf-8'))

    @classmethod
    def load(cls, path):
        """Load the Embedding object from a given zip path."""
        with zipfile.ZipFile(path, 'r') as archive:
        
            # load keys
            with archive.open('data.json') as f:
                data = json.loads(f.read().decode('utf-8'))
                keys = data.pop("_keys")
            
            # Load numpy array
            features = {}
            for key in keys:
                with archive.open(f'{key}.npy') as f:
                    features[key] = np.load(f)

        return cls(features=features, **data)

# class OnsetConditioner(WaveformConditioner):

#     def __init__(self, 
#             hop_length: int,
#             sample_rate: int,
#             threshold: float = 0.4, 
#             output_dim: int = 512, 
#         ):
#         import librosa
#         import madmom
#         from madmom.features.onsets import RNNOnsetProcessor, OnsetPeakPickingProcessor
#         import tempfile
#         import numpy as np 
#         super().__init__(dim=1, output_dim=output_dim)
#         self.hop_length = hop_length
#         self.threshold = threshold

    
#         self.proc = RNNOnsetProcessor(online=False)
#         self.onsetproc = OnsetPeakPickingProcessor(threshold=0.3,
#                                             fps=sample_rate/hop_length)
        
#     @torch.inference_mode()
#     def embed(self, audio: Tensor):
#         with tempfile.NamedTemporaryFile(suffix='.wav') as f:

#             act = proc(f.name)
#             onset_times = onsetproc(act)

#             # convert to indices for z array
#             onset_indices = librosa.time_to_frames(onset_times, sr=sig.sample_rate, hop_length=self.hop_length)

#             breakpoint()



REGISTRY = {
    "chroma": ChromaStemConditioner,
    "yamnet": YamnetConditioner,
}

