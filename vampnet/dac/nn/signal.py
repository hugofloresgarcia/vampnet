"""
add-ons to AudioSignal
"""
import math
from pathlib import Path
import warnings

import torch 
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import audiotools as at
from einops import rearrange

import math
import torch
import torch.nn as nn


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~ Signal                 ~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Signal(at.AudioSignal):
    """Signal is an extension of AudioSignal that adds additional
    functionality. It is meant to be a drop-in replacement. 
    I'm slowly building out the functionality of this class to 
    completely replace the AudioSignal class. 
    
    The main difference is that Signal objects are meant to be
    more than just audio signals, so that control signals can 
    be passed through similar pipelines (e.g. filters) as audio.
    It would probably make more sense to split stuff up into
    a Signal, AudioSignal, ControlSignal, Signal2D, etc. 
    But this was the quickest way to get things up and running for sketch2sound.
    """

    def __init__(
        self,
        audio_path_or_array: torch.Tensor | str | Path | np.ndarray,
        sample_rate: int = None,
        stft_params: at.STFTParams = None,
        offset: float = 0,
        duration: float = None,
        device: str = None,
        name: str = "",
        mask: torch.Tensor = None,
    ):
        audio_path = None
        audio_array = None

        if isinstance(audio_path_or_array, str):
            audio_path = audio_path_or_array
        elif isinstance(audio_path_or_array, Path):
            audio_path = audio_path_or_array
        elif isinstance(audio_path_or_array, np.ndarray):
            audio_array = audio_path_or_array
        elif torch.is_tensor(audio_path_or_array):
            audio_array = audio_path_or_array
        else:
            raise ValueError(
                f"audio_path_or_array must be either a Path, string, numpy array, or torch Tensor, but got {type(audio_path_or_array)}!"
            )

        self.path_to_file = None

        self.audio_data = None
        self.sources = None  # List of AudioSignal objects.
        self.stft_data = None
        if audio_path is not None:
            self.load_from_file(
                audio_path, offset=offset, duration=duration, device=device
            )
        elif audio_array is not None:
            assert sample_rate is not None, "Must set sample rate!"
            self.load_from_array(audio_array, sample_rate, device=device)

        self.window = None
        self.stft_params = stft_params


        self.metadata = {
            "name": name,
            "offset": offset,
            "duration": duration,
        }

        return


    def view(self,):
        return Signal(self.samples, self.sample_rate, name=self.name)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~ getters // setters ~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @property
    def name(self):
        return self.metadata["name"]
    

    @name.setter
    def name(self, value):
        self.metadata["name"] = value


    @property
    def audio_data(self):
        """Returns the audio data tensor in the object.

        Audio data is always of the shape
        (batch_size, num_channels, num_samples). If value has less
        than 3 dims (e.g. is (num_channels, num_samples)), then it will
        be reshaped to (1, num_channels, num_samples) - a batch size of 1.

        Parameters
        ----------
        data : typing.Union[torch.Tensor, np.ndarray]
            Audio data to set.

        Returns
        -------
        torch.Tensor
            Audio samples.
        """
        return self._audio_data

    # OVERRIDE
    @audio_data.setter
    def audio_data(self, data: torch.Tensor | np.ndarray):
        if data is not None:
            assert torch.is_tensor(data), "audio_data should be torch.Tensor"
            assert data.ndim == 3, "audio_data should be 3-dim (B, C, T)"
        self._audio_data = data
        # Old loudness value not guaranteed to be right, reset it.
        self._loudness = None
        self.stft_data = None
        return
    
    @property
    def d(self, ):
        return self.audio_data

    @d.setter
    def d(self, data):
        self.audio_data = data
        return

    # Indexing
    def __getitem__(self, key):
        if torch.is_tensor(key) and key.ndim == 0 and key.item() is True:
            assert self.batch_size == 1
            audio_data = self.audio_data
            _loudness = self._loudness
            stft_data = self.stft_data

        elif isinstance(key, (bool, int, list, slice, tuple)) or (
            torch.is_tensor(key) and key.ndim <= 1
        ):
            # Indexing only on the batch dimension.
            # Then let's copy over relevant stuff.
            # Future work: make this work for time-indexing
            # as well, using the hop length.
            audio_data = self.audio_data[key]
            # _loudness = self._loudness[key] if self._loudness is not None else None
            # stft_data = self.stft_data[key] if self.stft_data is not None else None


        copy = type(self)(audio_data, self.sample_rate, stft_params=self.stft_params)
        copy._loudness = None
        copy.stft_data = None
        copy.sources = None

        return copy


    def __setitem__(self, key, value):
        if not isinstance(value, type(self)):
            self.audio_data[key] = value
            return

        if torch.is_tensor(key) and key.ndim == 0 and key.item() is True:
            assert self.batch_size == 1
            self.audio_data = value.audio_data
            self._loudness = value._loudness
            self.stft_data = value.stft_data
            return

        elif isinstance(key, (bool, int, list, slice, tuple)) or (
            torch.is_tensor(key) and key.ndim <= 1
        ):
            if self.audio_data is not None and value.audio_data is not None:
                self.audio_data[key] = value.audio_data
            if self._loudness is not None and value._loudness is not None:
                self._loudness[key] = value._loudness
            if self.stft_data is not None and value.stft_data is not None:
                self.stft_data[key] = value.stft_data
            return

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~ stft ~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @property
    def stft_data(self):
        """Returns the STFT data inside the signal. Shape is
        (batch, channels, frequencies, time).

        Returns
        -------
        torch.Tensor
            Complex spectrogram data.
        """
        return self._stft_data

    @stft_data.setter
    def stft_data(self, data: torch.Tensor | np.ndarray):
        if data is not None:
            assert torch.is_tensor(data) and torch.is_complex(data)
            if self.stft_data is not None and self.stft_data.shape != data.shape:
                warnings.warn("stft_data changed shape")
            if data.ndim != 4:
                raise ValueError("stft_data should be 4-dim (B, C, F, T)")
        # print(f"setting stft_data to {(data.shape if data is not None else None)}")
        self._stft_data = data
        return
    

    def stft(
        self,
        window_length: int = None,
        hop_length: int = None,
        window_type: str = None,
        match_stride: bool = None,
        padding_type: str = None,
    ):
        """Computes the short-time Fourier transform of the audio data,
        with specified STFT parameters.

        Parameters
        ----------
        window_length : int, optional
            Window length of STFT, by default ``0.032 * self.sample_rate``.
        hop_length : int, optional
            Hop length of STFT, by default ``window_length // 4``.
        window_type : str, optional
            Type of window to use, by default ``sqrt\_hann``.
        match_stride : bool, optional
            Whether to match the stride of convolutional layers, by default False
        padding_type : str, optional
            Type of padding to use, by default 'reflect'

        Returns
        -------
        torch.Tensor
            STFT of audio data.

        Examples
        --------
        Compute the STFT of an AudioSignal:

        >>> signal = AudioSignal(torch.randn(44100), 44100)
        >>> signal.stft()

        Vary the window and hop length:

        >>> stft_params = [STFTParams(128, 32), STFTParams(512, 128)]
        >>> for stft_param in stft_params:
        >>>     signal.stft_params = stft_params
        >>>     signal.stft()

        """
        window_length = (
            self.stft_params.window_length
            if window_length is None
            else int(window_length)
        )
        hop_length = (
            self.stft_params.hop_length if hop_length is None else int(hop_length)
        )
        window_type = (
            self.stft_params.window_type if window_type is None else window_type
        )
        match_stride = (
            self.stft_params.match_stride if match_stride is None else match_stride
        )
        padding_type = (
            self.stft_params.padding_type if padding_type is None else padding_type
        )

        window = self.get_window(window_type, window_length, self.audio_data.device)
        window = window.to(self.audio_data.device)

        audio_data = self.audio_data
        right_pad, pad = self.compute_stft_padding(
            window_length, hop_length, match_stride
        )
        audio_data = torch.nn.functional.pad(
            audio_data, (pad, pad + right_pad), padding_type
        )
        stft_data = torch.stft(
            audio_data.reshape(-1, audio_data.shape[-1]),
            n_fft=window_length,
            hop_length=hop_length,
            window=window,
            return_complex=True,
            center=True,
        )
        _, nf, nt = stft_data.shape
        stft_data = stft_data.reshape(self.batch_size, self.num_channels, nf, nt)

        if match_stride:
            # Drop first two and last two frames, which are added
            # because of padding. Now num_frames * hop_length = num_samples.
            stft_data = stft_data[..., 2:-2]
        self.stft_data = stft_data

        return stft_data
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~  helpers ~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @staticmethod
    def wave( 
        shape="saw",
        frequency: float = 440.0,
        phase: float = 0.0,
        duration: float = 1.0,
        sample_rate: int = 44100,
    ) -> "Signal":

        t = torch.linspace(0, duration, int(duration * sample_rate))
        if shape == "sine":
            data =  torch.sin(2 * np.pi * frequency * t + phase)
        elif shape == "saw":
            data = 2 * (t * frequency - torch.floor(t * frequency + 0.5))
        elif shape == "square":
            data = 2 * (torch.floor(2 * frequency * t) % 2) - 1
        elif shape == "triangle":
            data = 2 * torch.abs(2 * (t * frequency - torch.floor(t * frequency + 0.5))) - 1
        else:
            raise ValueError(f"Unknown waveform shape: {shape}")
    
        return Signal(
            data.unsqueeze(0).unsqueeze(0),
            sample_rate,
            name=f"{shape}_waveform"
        )
  
        
    def crop(
        self, start_s: float, end_s: float
    ):
        start_frame = int(start_s * self.sample_rate)
        end_frame = int(end_s * self.sample_rate)
        return Signal(
            self.samples[:, :, start_frame:end_frame],
            self.sample_rate,
            name=f"{self.name}_crop={start_s}-{end_s}"
        )

    def ensure_duration(
        self, min_duration: float = None, max_duration: float = None
    ):
        sig = self.view()
        if min_duration is not None:
            if sig.duration < min_duration:
                sig.samples = torch.nn.functional.pad(
                    sig.samples, (0, int(min_duration * sig.sample_rate) - sig.num_frames)
                )
        if max_duration is not None:
            if sig.duration > max_duration:
                return sig.crop(0, max_duration)
        return sig

    @property
    def time(self,) -> torch.Tensor:
        return torch.arange(self.num_frames) / self.sample_rate
    
    @property
    def num_frames(self,) -> int:
        return self.samples.shape[-1]

    @property
    def stft_frame_rate(self):
        """Computes the frame rate of the STFT.
        """
        return self.sample_rate / self.stft_params.hop_length
    

    def fft_frequencies(self):
        """Computes the FFT frequencies of the audio data.
        """
        return torch.fft.rfftfreq(
            n=self.stft_params.window_length, 
            d=1/self.sample_rate
        ).to(self.samples)
    

    def db2pow(self):
        self = self.view()
        self.d = db2pow(self.samples)
        return self


    def pow2db(self):
        self = self.view()
        self.d = pow2db(self.samples)
        return self


    def amp2db(self):
        self = self.view()
        self.d = amp2db(self.samples)
        return self


    def db2amp(self):
        self = self.view()
        self.d = db2amp(self.samples)
        return self


    # ~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~ filters !!!!! ~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~
    def low_pass(self, 
        cutoffs: torch.Tensor | float, 
        zeros: int = 51, 
        filtfilt: bool = False
    ) -> "Signal":
        cutoffs = at.util.ensure_tensor(cutoffs)

        if filtfilt:
            sig = at.AudioSignal(self.samples, self.sample_rate).low_pass(cutoffs / 2, zeros)
            rev = Signal(sig.samples.flip(-1), sig.sample_rate)
            rev = at.AudioSignal(rev.samples, rev.sample_rate).low_pass(cutoffs / 2, zeros)
            sig = Signal(rev.samples.flip(-1), rev.sample_rate)
        else:
            # import time
            # start = time./time()
            sig = at.AudioSignal(self.samples, self.sample_rate).low_pass(cutoffs, zeros)
            sig = Signal(sig.samples, sig.sample_rate)

        if cutoffs.ndim == 0:
            cutoffs = cutoffs.unsqueeze(0)
        sig.name = f"{self.name}-lowpass-{cutoffs[0]}Hz".strip("-")
        return sig


    def high_pass(
        self, cutoffs: torch.Tensor | np.ndarray | float, zeros: int = 51
    ):
        """High-passes the signal in-place. Each item in the batch
        can have a different high-pass cutoff, if the input
        to this signal is an array or tensor. If a float, all
        items are given the same high-pass filter.

        Parameters
        ----------
        cutoffs : typing.Union[torch.Tensor, np.ndarray, float]
            Cutoff in Hz of high-pass filter.
        zeros : int, optional
            Number of taps to use in high-pass filter, by default 51

        Returns
        -------
        AudioSignal
            High-passed AudioSignal.
        """
        import julius
        self = self.view()
        cutoffs = at.util.ensure_tensor(cutoffs, 2, self.batch_size)
        cutoffs = cutoffs / self.sample_rate
        filtered = torch.empty_like(self.audio_data)

        for i, cutoff in enumerate(cutoffs):
            hp_filter = julius.HighPassFilter(cutoff.cpu(), zeros=zeros).to(self.device)
            filtered[i] = hp_filter(self.audio_data[i])

        self.audio_data = filtered
        self.stft_data = None
        return self
    
    
    
    def median_filter(self, kernel_size: int | torch.Tensor = 3) -> "Signal":
        sig = self.view()
        if not torch.is_tensor(kernel_size):
            assert isinstance(kernel_size, int), "kernel_size must be an int or a tensor"
            kernel_size = torch.tensor([kernel_size] * self.batch_size)
        else:
            if kernel_size.ndim == 0:
                kernel_size = kernel_size.unsqueeze(0)

        assert kernel_size.numel() == self.batch_size
        for i, ks in enumerate(kernel_size):
            sig.samples[i] = MedianPool2d((1, int(ks)), stride=1, same=True)(sig.samples[i].unsqueeze(0).unsqueeze(0)).squeeze(0)
        
        sig.name = f"{self.name}_median_filter".strip("-")
        return sig

    # ~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~ descriptors!!!!! ~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~

    def spectrogram(self, power: int = 1, a_weighted=False) -> torch.Tensor:
        """Computes the spectrogram of the audio data.
        """        
        
        if a_weighted:
            import librosa
            a_weighting = librosa.A_weighting(self.fft_frequencies().cpu().numpy())
            a_weighting = torch.from_numpy(a_weighting).to(self.samples)
            a_weighting = db2pow(a_weighting)
            return (self.magnitude * a_weighting[None, None, :, None]) ** power
        else:
            return self.magnitude ** power
            
    
    def rms(self, a_weighted=True) -> "Signal":
        """Computes the root mean square of the audio data.
        ported from librosa.feature.rms (STFT implementation)

        Returns
        -------
        torch.Tensor
            RMS of audio data.
        """
        if self.stft_data is None:
            self.stft()
        
        # Check the frame length
        frame_length = self.stft_params.window_length

        # power spectrogram
        x = self.spectrogram(power=2, a_weighted=a_weighted)

        # Adjust the DC and sr/2 component
        x[..., 0, :] *= 0.5
        if frame_length % 2 == 0:
            x[..., -1, :] *= 0.5

        # Calculate power
        power = 2 * torch.sum(x, dim=-2, keepdim=False) / frame_length**2
        return Signal(
            torch.sqrt(power), 
            self.stft_frame_rate, 
            name=f"{self.name}-rms".strip("-")
        )
    

    def spectral_centroid(self) -> "Signal":
        """Computes the spectral centroid of the audio data.
        """
        x = self.spectrogram(power=1)

        # normalize the spectrogram
        x = x / (x.sum(dim=-2, keepdim=True) + 1e-10) 

        # get the frequency bins
        freq_bins = self.fft_frequencies()

        return Signal(
            torch.sum(freq_bins[None, None, :, None] * x, dim=-2), 
            self.stft_frame_rate, 
            name=f"{self.name}-spectral_centroid".strip("-")
        )


    def spectral_bandwidth(self, p: int = 2) -> "Signal":
        """Computes the spectral bandwidth of the audio data.
        """
        x = self.magnitude

        # normalize the spectrogram 
        #TODO: hugo, you just inserted this here, but 
        # i'm not sure if this is the right kind of norm to do here
        # can we double check against the librosa source? 
        x = x / (x.sum(dim=-2, keepdim=True) + 1e-10) 

        # get the frequency bins
        freq_bins = self.fft_frequencies()
        deviation = torch.abs(freq_bins[None, None, :, None] - self.spectral_centroid().d[:, :, None, :])

        bw: torch.Tensor = torch.sum(x * deviation**p, dim=-2, keepdim=False) ** (1/p)
        return Signal(
            bw, 
            self.stft_frame_rate, 
            name=f"{self.name}-spectral_bandwidth".strip("-")
        )
    
    
    def spectral_flatness(self, power=2.0) -> "Signal":
        x = self.magnitude

        thresh = torch.maximum(torch.full_like(x, 1e-10), x ** power)
        geomean = torch.exp(torch.mean(torch.log(thresh), dim=-2, keepdim=False))
        amean = torch.mean(thresh, dim=-2, keepdim=False) 
        flatness = geomean / amean
        return Signal(
            flatness, 
            self.stft_frame_rate, 
            name="spectral_flatness"
        )


    def spectral_flatness_db(self, power=2.0) -> "Signal":
        flatness = self.spectral_flatness(power)
        return Signal(
            amp2db(flatness.d),
            flatness.sample_rate,
            name="spectral_flatness_db"
        )


    def torchcrepe_pp(self, decode="argmax", probabilities=None) -> tuple[torch.Tensor, torch.Tensor]:
        import torchcrepe
        
        assert self.num_channels == 1, "only mono signals supported (TODO)"
        if probabilities is None:
            probabilities = self.crepe_ppg(_raw=True).clone()
        else: 
            probabilities = probabilities.clone()
            
        nb = self.d.shape[0]

        if decode == "argmax":
            decoder = torchcrepe.decode.argmax
        elif decode == "weighted_argmax":
            decoder = torchcrepe.decode.weighted_argmax
        elif decode == "viterbi":
            decoder = torchcrepe.decode.viterbi

        result = torchcrepe.postprocess(probabilities,
                    50.0,
                    torchcrepe.MAX_FMAX,
                    decoder,
                    return_harmonicity=True,
                    return_periodicity=True)
        
        if isinstance(result, tuple):
            result = (result[0].to(self.d.device),
                        result[1].to(self.d.device))
        else:
            result = result.to(self.d.device)

        pitch, periodicity = result
        
        pitch = rearrange(pitch, "1 (b t) -> b 1 t", b=nb)
        periodicity = rearrange(periodicity, "1 (b t) -> b 1 t", b=nb)


        # interpolate back to stft frame rate
        stft_frames = int(math.ceil(self.stft_frame_rate * self.duration))
        pitch = torch.nn.functional.interpolate(
            pitch, 
            size=stft_frames, 
            mode="nearest"
        )
        periodicity = torch.nn.functional.interpolate(
            periodicity, 
            size=stft_frames, 
            mode="nearest"
        )


        return pitch, periodicity
 

    def crepe_ppg(self, _raw=False) -> torch.Tensor:
        import time
        import torchcrepe
        t0 = time.time()
        assert self.num_channels == 1, "only batch 1 mono signals supported (TODO)"

        # THIS IS MUCH FASTER THAN torchcrepe.preprocess!
        # make a sig clone
        sig = self.clone()
        sig.resample(torchcrepe.SAMPLE_RATE)
        hop_length = torchcrepe.SAMPLE_RATE // 100
        
        audio = sig.samples
        nb = audio.shape[0]
        nc = audio.shape[1]

        # pad!
        total_frames = 1 + int(audio.shape[-1] // hop_length)
        audio = torch.nn.functional.pad(
            audio, 
            (torchcrepe.WINDOW_SIZE // 2, torchcrepe.WINDOW_SIZE // 2),
        )

        # unfold into frames
        frames = torch.nn.functional.unfold(
            audio[:, :, None, :],
            kernel_size=(1, torchcrepe.WINDOW_SIZE),
            stride=(1, hop_length),
        )

        # collapse batch dim
        frames = rearrange(frames, "b f t -> (b t) f")

        # Mean-center
        frames = frames - frames.mean(dim=-1, keepdim=True)
        frames = frames / torch.max(
            torch.tensor(1e-10, device=frames.device),
            frames.std(dim=-1, keepdim=True)
        )

        probabilities = torchcrepe.infer(frames, 
            device=self.samples.device, model='tiny',
        )
        probabilities = rearrange(probabilities, "t f -> 1 f t")

        # Sampling is non-differentiable, so remove from graph
        # probabilities = probabilities.detach()

        if _raw: return probabilities

        # Convert frequency range to pitch bin range
        fmin = 50.
        fmax = 2006.
        minidx = torchcrepe.convert.frequency_to_bins(torch.tensor(fmin))
        maxidx = torchcrepe.convert.frequency_to_bins(torch.tensor(fmax),
                                                    torch.ceil)

        # Remove frequencies outside of allowable range
        probabilities[:, :minidx, :] = 0.0 # HUGO: changed it from -inf to 0.0
        probabilities[:, maxidx:, :] = 0.0 # HUGO: changed it from -inf to 0.0

        # interpolate the probabilities to our stft frame rate
        stft_frames = int(math.ceil(self.stft_frame_rate * self.duration))

        # take out the batch and channels
        probabilities = rearrange(probabilities, "1 f (b c k) -> (b c) f k", b=nb, c=nc)

        probabilities = torch.nn.functional.interpolate(
            probabilities, 
            size=stft_frames, 
            mode="nearest"
        )
        # print(f"post processing took {time.time() - t0:.5f}s")
        # print(probabilities.shape)
        return probabilities


    def crepe_ppg_bins_to_frequency(self, bins: torch.Tensor) -> torch.Tensor:
        import torchcrepe
        return torchcrepe.convert.bins_to_frequency(bins)


    def pitch_periodicity(self, method="torchcrepe") -> tuple["Signal", "Signal"]:
        if method == "torchcrepe":
            pitch, periodicity = self.torchcrepe_pp()
        else:
            raise ValueError(f"Unknown method: {method}")

        new_framerate = pitch.shape[-1] / self.duration
        
        # concatenate the pitch and periodicity
        pitch = Signal(
            pitch, 
            new_framerate, 
            name=f"{self.name}_pitch".strip("-")
        )
        periodicity = Signal(
            periodicity,
            new_framerate,
            name=f"{self.name}_periodicity".strip("-")
        )
        return pitch, periodicity
    

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~ viz helpers!!!!!!!!!!!! ~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def plot(self, 
            ax=None, 
            batch_idx=0,
            chan_idx=0,
            **kwargs
        ):
        """Plot a time-varying 1-D descriptor of the audio data.
        
        Parameters
        ----------
        data : torch.Tensor
            The descriptor data to plot. Shape (batch, channel, frames)
        ax : plt.Axes, optional
            The axis to plot the RMS on. If None, a new figure is created.
        stem : bool, optional
            Whether to plot the RMS as a stem plot.
        **kwargs   
            Additional keyword arguments to pass to ax.plot or ax.stem.
        """
        data = self.d
        assert self.d.ndim == 3, "data must have shape (batch, channel, frames)"
        assert self.d.shape[-2] == 1, "data must be 1d to use plot_descriptor. use plot_"

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        handle,  = ax.plot(self.time, data[batch_idx][chan_idx], **kwargs)

        # ax.set_title(self.name)
        # plt.tight_layout()

        return handle


    def plot_spec(self, 
        log_magnitude=True,
        ax=None,
        spec=None, 
        **kwargs
    ):
        spec = spec if spec is not None else self.spectrogram()
        if log_magnitude:
            spec = 10 * torch.log10(spec + 1e-10)

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        import librosa
        cmap = kwargs.pop("cmap", "viridis")
        y_axis = kwargs.pop("y_axis", "log")
        librosa.display.specshow(
            spec[0][0].detach().cpu().numpy(), 
            sr=self.sample_rate, 
            hop_length=self.stft_params.hop_length, 
            x_axis="time", 
            y_axis=y_axis, 
            ax=ax, 
            cmap=cmap, 
            **kwargs
        )
        ax.set_xlabel("")

        return fig, ax
  
  
    def visualize(self,
        rms=None,
        pitch=False,
        ppg=True,
        **kwargs
    ) -> "Signal":
        plt.clf()
        fig, ax = plt.subplots(2, 1, gridspec_kw={"height_ratios": [1, 1]}, sharex=True)
        if not isinstance(ax, np.ndarray):
            ax = [ax]

        # plot the spectrogram
        spax = ax[0]
        self.plot_spec(ax=spax,cmap="viridis")

        #  RMS!!!!!!!!!!!!!
        if rms is None:
            rms = self.rms()
        else:
            print(f"using provided rms")
        rmsax = ax[0].twinx()
        rmsax.grid(False)
        rmsax.set_ylim(-0.01, 0.7)
        rms.plot(ax=rmsax, color="black", label="rms")
        rmsax.set_title("")

        # CENTROID !!!!!!!!!
        centroid = self.spectral_centroid()
        centroid.plot(
            ax=spax,    
            markersize=3, color="magenta", label="centroid (brightness)",
        )
        spax.grid(False)
        spax.set_title("")

        # PITCH 
        plot_pitch = True
        if plot_pitch:
            pitch, periodicity = self.pitch_periodicity(method='torchcrepe')
            spax.scatter(
                pitch.time, pitch.d[0][0], 
                label="pitch (Hz)", color="orangered", alpha=torch.clamp(periodicity.d[0][0], 0.99, 1.0), 
                s=3
            )
            # median filter
            # pitch = pitch.median_filter(kernel_size=50)

            # apply a median filter to the pitch
            # replot the pitch
            spax.scatter(
                pitch.time, pitch.d[0][0], 
                label="pitch (Hz)", color="blue", alpha=torch.clamp(periodicity.d[0][0], 0.99, 1.0), 
                s=3
            )
            # plot the pitch line,
            # set the opacity to whatever the periodicity is
            # spax.scatter(
            #     pitch.time, pitch.d[0][0], 
            #     label="pitch (Hz)", color="orangered", alpha=torch.clamp(periodicity.d[0][0], 0.0, 1.0), 
            #     s=3
            # )

        # BANDWIDTH
        bw = self.spectral_bandwidth()
        spax.fill_between(
            bw.time, 
            torch.maximum(
                centroid.d - (bw.d / 2), 
                torch.full_like(centroid.d, 0)
            )[0][0],
            torch.minimum(
                centroid.d + (bw.d / 2), 
                torch.full_like(centroid.d, self.sample_rate / 2)
            )[0][0],
            label="bandwidth", alpha=0.5, color="slategrey"
        )


        # FLATNESS!!!!!
        flatness = self.spectral_flatness_db()
        flatax = ax[0].twinx()
        flatax.spines["right"].set_position(("axes", 1.2))
        flatplot, = flatax.plot(flatness.time, flatness.d[0][0], color="slategray", alpha=1.0, label="flatness (dB)")
        flatax.yaxis.label.set_color(flatplot.get_color())
        flatax.grid(False)
        flatax.set_ylim(-150, 0)


        # ZERO CROSSING RATE
        plot_zcr = False
        if plot_zcr:
            zcr = self.librosa_zero_crossing_rate()
            zcrax = ax[0].twinx()
            zcrplot, = zcrax.plot(zcr.time, zcr.d[0][0], color="dodgerblue", alpha=1.0, label="zero crossing rate")
            zcrax.set_frame_on(True)
            zcrax.spines["right"].set_position(("axes", 1.4))
            zcrax.spines["right"].set_visible(True)
            zcrax.spines["right"].set_edgecolor(zcrplot.get_color())
            zcrax.yaxis.label.set_color(zcrplot.get_color())
            zcrax.grid(False)
            # zcrax.set_ylim(0, 0.5)
    
        ppgax = ax[1]
        if ppg:
            # on the 3rd axis, plot the crepe ppg
            # HUGO (TODO): evidence we need a separate class for "control signals"
            ppg = Signal(self.crepe_ppg(), sample_rate=self.stft_frame_rate)

            x_coords = ppg.time
            y_coords = ppg.crepe_ppg_bins_to_frequency(torch.arange(ppg.d.shape[-2]))
            ppgd = ppg.d
            # set the alpha to 0.0 everywhere where the ppg is < 0.4
            thr = 0.0001
            # alphas = torch.where(ppg.d < thr, torch.zeros_like(ppg.d), ppg.d)
            # alphas = torch.ones_like(ppg.d)[:1]
            # ppgd = torch.where(ppg.d < thr, torch.zeros_like(ppg.d), torch.ones_like(ppg.d))

            self.plot_spec(
                ax=ppgax, 
                spec=ppgd.unsqueeze(0), 
                x_coords=x_coords, 
                y_coords=y_coords, 
                # alpha=alphas, 
                cmap="Reds"
            )

        harmonic_map = False
        if harmonic_map:
            harm_map = self.harmonics_map()
            # ax[1].clear()
            self.plot_spec(ax=ax[1], spec=harm_map, cmap="viridis", alpha=1, y_axis="mel", fmin=40, fmax=2000)

        # fig.subplots_adjust(hspace=0.05)
        fig.legend(loc="upper right")
        return fig, ax


    def visualize_video(
        self, 
        output_path: str = "output.mp4",
        fps=30, 
        window_duration: int = 3.0, 
        **kwargs
    ):
        import matplotlib.style as mplstyle
        mplstyle.use('fast')
        window_duration = self.duration
        # write the audio file
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdirname:
            assert self.shape[0] == 1, "only batch 1 signals supported"
            # turn the dpi down to make it faster
            dpi = 50
            szi = (16, 10)
            plt.rcParams["figure.dpi"] = dpi
            # set the fig to a 9:16 aspect 
            plt.rcParams["figure.figsize"] = szi
            size = (szi[0] * dpi, szi[1] * dpi)

            figs = []

            playhead_pos = 0
            playhead_step = 1 / fps
            fig, ax = self.visualize(**kwargs, animated=True)

            _ax = ax[1]

            # make it so that the image will begin exactly at x=0.0
            plt.margins(x=0)
            line = _ax.axvline(playhead_pos, color="red", alpha=1.0, linewidth=5)

            i = 0 
            figs = []
            while playhead_pos < self.duration:

                print("rendering frame at", playhead_pos)
                line.set_xdata([playhead_pos, playhead_pos])

                plt2pil(False).save(f"{tmpdirname}/{i}.png")
                figs.append(f"{tmpdirname}/{i}.png")

                i += 1
                playhead_pos += playhead_step


            # get the size of the first figure
            plt.close("all")

            audio_out = f"{tmpdirname}/audio.wav"
            self.write(audio_out)

            # Good old ffmpeg
            command = ['ffmpeg', '-y']
            command += ['-loglevel', 'panic']
            # if seek is not None:
                # command += ['-ss', str(seek)]
            command += ['-i', audio_out]
            # if duration is not None:
                # command += ['-t', str(duration)]
            command += ['-f', 'f32le']
            command += ['-']

            import subprocess as sp
            audio_cmd = []
            audio_cmd += ["-i", str(Path(audio_out).resolve())]
            command = [
                "ffmpeg", "-y", 
                "-loglevel", "panic", 
                "-r", str(fps), 
                "-f", "image2", 
                "-s", f"{size[0]}x{size[1]}", 
                "-i", "%d.png"
            ] + audio_cmd + [
                "-c:a", "libvorbis",
                "-vcodec", "libx264", 
                "-crf", "10", 
                "-pix_fmt", "yuv420p",
                str(Path(output_path).resolve())
            ]
            print(f"running: {' '.join(command)}")
            print(sp.run(command,check=True,cwd=tmpdirname))
        


if __name__ == "__main__":

    sig = Signal("scratch/car.wav")
    sig.visualize()
    img0 = plt2pil()

    vertical_pil_stack([img0,]).save("test.png")

    sig.write("test.wav")
    # sig.visualize_video("test.mp4", window_duration=3.0)
    

