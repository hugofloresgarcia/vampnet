"""
add-ons to AudioSignal
"""
import math
from pathlib import Path

import torch 
from torch import nn
import numpy as np
import torchaudio
import matplotlib.pyplot as plt
from PIL import Image

import audiotools as at


def set_plt_params():
    """ Initialize a wide-ish figure, ideal for plotting
    descriptor signals

    Parameters
    ----------
    **kwargs
        Additional keyword arguments to pass to plt.subplots
    """
    plt.style.use("fivethirtyeight")
    plt.rcParams.update({
        "lines.linewidth": 0.8, 
        "figure.dpi": 120,
        "figure.figsize": (6, 2.75),
        "axes.labelsize": 7,
        "xtick.labelsize": 4,
        "ytick.labelsize": 4,
        "figure.titlesize": 12,
        "axes.grid": True,
        "axes.titlesize": 10,
        "axes.labelsize": 8,
        "grid.color": "#f0f0f0",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.5,
        "grid.linestyle": ":",
        "legend.fontsize": 7,
        # "savefig.facecolor":  (0.0, 0.0, 0.0, 0.0),
        # "savefig.edgecolor":  (0.0, 0.0, 0.0, 0.0),
        # "axes.edgecolor":  (0.0, 0.0, 0.0, 0.0),
        # "figure.edgecolor":  (0.0, 0.0, 0.0, 0.0),
    })

set_plt_params()


def frame_to_ms(frame, frame_rate):
    """Converts an STFT frame index to milliseconds.
    """
    return (frame / frame_rate) * 1000


def plt2pil():
    """Converts a matplotlib figure to a PIL image.
    """
    from io import BytesIO
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.clf()
    return Image.open(buf)


def vertical_pil_stack(imgs):
    """Stacks PIL images vertically.
    """
    widths, heights = zip(*(i.size for i in imgs))
    total_width = max(widths)
    total_height = sum(heights)

    new_img = Image.new("RGB", (total_width, total_height))
    y_offset = 0
    for img in imgs:
        new_img.paste(img, (0, y_offset))
        y_offset += img.size[1]

    return new_img

def horizontal_pil_stack(imgs):
    """Stacks PIL images horizontally.
    """
    widths, heights = zip(*(i.size for i in imgs))
    total_width = sum(widths)
    total_height = max(heights)

    new_img = Image.new("RGB", (total_width, total_height))
    x_offset = 0
    for img in imgs:
        new_img.paste(img, (x_offset, 0))
        x_offset += img.size[0]

    return new_img

class Signal(at.AudioSignal):


    def __init__(
        self,
        audio_path_or_array: torch.Tensor | str | Path | np.ndarray,
        sample_rate: int = None,
        stft_params: at.STFTParams = None,
        offset: float = 0,
        duration: float = None,
        device: str = None,
        name: str = "",
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
                "audio_path_or_array must be either a Path, "
                "string, numpy array, or torch Tensor!"
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

        self.descriptors = {}

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
        # clear descriptors
        self.descriptors = {}
        return


    @property
    def d(self, ):
        return self.audio_data

    @d.setter
    def d(self, data):
        self.audio_data = data
        return

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~ sugar // helpers ~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @property
    def time(self,) -> torch.Tensor:
        return torch.arange(self.num_frames) / self.sample_rate
    

    @property
    def num_frames(self,) -> int:
        return self.samples.shape[-1]


    def apply_envelope(self, attack: float = 0.01):
        # make an envelope that spans the duration 
        # of the audio sig and has the attack time
        # specified

        t = torch.arange(0, self.duration, 1/self.sample_rate)

        # make the envelope
        envelope = torch.ones_like(t)

        # attack
        attack_frames = int(attack * self.sample_rate)
        envelope[:attack_frames] = torch.linspace(0, 1, attack_frames)
        
        # decay for the rest of the signal
        envelope[attack_frames:] = torch.linspace(1, 0, len(envelope) - attack_frames)

        # apply the envelope
        self.samples = self.samples * envelope[None, None, :]
        return self


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
        )
    
    
    @staticmethod
    def pow2db(x):
        return 10 * torch.log10(x + 1e-10)

    @staticmethod
    def db2pow(x):
        return 10**(x / 10)

    # ~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~ effectsssss!!!!! ~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~
    def low_pass_decimate(self, cutoff: torch.Tensor) -> "Signal":
        """Decimates the audio data by a factor.
        """
        out = self.clone()
        # calculate the scale factor based on the sample rate etc
        scale_factor = cutoff / self.sample_rate
        print(f"decimating by {scale_factor}")

        num_frames = self.num_frames
        out.samples = torch.nn.functional.interpolate(
            out.samples, scale_factor=scale_factor, mode="nearest"
        )

        # interpolate back up to the original length
        out.samples = torch.nn.functional.interpolate(
            out.samples, size=num_frames, mode="nearest"
        )

        return out


    def low_pass(self, 
        cutoffs: torch.Tensor | float, 
        zeros: int = 51, 
        filtfilt: bool = False
    ) -> "Signal":
        
        cutoffs = at.util.ensure_tensor(cutoffs)

        if filtfilt:
            # import time
            # start = time.time()
            sig = at.AudioSignal(self.samples, self.sample_rate).low_pass(cutoffs / 2, zeros)
            rev = Signal(sig.samples.flip(-1), sig.sample_rate)
            rev = at.AudioSignal(rev.samples, rev.sample_rate).low_pass(cutoffs / 2, zeros)
            sig = Signal(rev.samples.flip(-1), rev.sample_rate)
            # print(f"filtfilt took {time.time() - start:.5f}s")
        else:
            # import time
            # start = time./time()
            sig = at.AudioSignal(self.samples, self.sample_rate).low_pass(cutoffs, zeros)
            sig = Signal(sig.samples, sig.sample_rate)
            # print(f"lowpass took {time.time() - start:.5f}s")

        if cutoffs.ndim == 0:
            cutoffs = cutoffs.unsqueeze(0)
        sig.name = f"{self.name}-lowpass-{cutoffs[0]}Hz".strip("-")
        return sig
    

    def savgol(self, window_length: int = 5, order: int = 3) -> "Signal":
        from audit.helpers.savgol import SavitzkyGolayFilter
        sg = SavitzkyGolayFilter(window_length, order)
        return Signal(
            sg(self.samples),
            self.sample_rate,
            name=f"{self.name}-savgol_W{window_length}_O{order}".strip("-")
        )


    def hilbert(self) -> "Signal":
        from audit.helpers.hilbert import HilbertTransform
        ht = HilbertTransform(axis=2)
        return Signal(
            ht(self.samples),
            self.sample_rate,
            name=f"{self.name}_hilbert".strip("-")
        )
        

    # ~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~ descriptors!!!!! ~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~

    def spectrogram(self, power: int = 1) -> torch.Tensor:
        """Computes the spectrogram of the audio data.
        """        
        return self.magnitude ** power


    def rms(self) -> "Signal":
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
        x = self.stft_data.real**2 + self.stft_data.imag**2

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
        x = self.magnitude

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

    def set_pitch_periodicity_method(self, method: str):
        assert method in ("penn", "torchcrepe"), "method must be one of ('penn', 'torchcrepe')"
        self.pitch_periodicity_method = method
        return self


    def _pitch_periodicity(self, ) -> "Signal":
        method = self.pitch_periodicity_method

        key = f"pp_{method}"
        if key in self.descriptors:
            return self.descriptors[key]
        else: 
            if method == "penn":
                import penn

                hopsize = self.stft_params.hop_length / self.sample_rate
                fmin = 40.
                fmax = 2000.

                assert self.samples.shape[1] == 1, "only mono signals supported"

                pitch, periodicity = penn.from_audio(
                    self.samples.squeeze(1), self.sample_rate,
                    hopsize=hopsize, 
                    fmin=fmin,
                    fmax=fmax,
                )
                pitch, periodicity = pitch[:, None, :], periodicity[:, None, :]
                self._penn = (pitch, periodicity)


            elif method == "torchcrepe":
                import torchcrepe
                pitch, periodicity = torchcrepe.predict(
                    self.samples.squeeze(1), self.sample_rate,
                    hop_length=self.stft_params.hop_length,
                    fmin=40,
                    fmax=2000,
                    return_periodicity=True
                )
                win_length = 3
                # Median filter noisy confidence value
                periodicity = torchcrepe.filter.median(periodicity, win_length)

                # Remove inharmonic regions
                pitch = torchcrepe.threshold.At(.21)(pitch, periodicity)

                # Optionally smooth pitch to remove quantization artifacts
                pitch = torchcrepe.filter.mean(pitch, win_length)
        
        self.descriptors[key] = (pitch, periodicity)
        return pitch, periodicity


    def pitch(self) -> "Signal":
        pitch, _ = self._pitch_periodicity()
        new_framerate = pitch.shape[-1] / self.duration
        return Signal(
            pitch,
            new_framerate,
            name=f"{self.name}_pitch".strip("-")
        )


    def periodicity(self) -> "Signal":
        _, periodicity = self._pitch_periodicity()
        new_framerate = periodicity.shape[-1] / self.duration
        return Signal(
            periodicity,
            new_framerate,
            name=f"{self.name}_periodicity".strip("-")
        )

    # ~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~ librosa features ~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~
    def librosa_centroid(self, ) -> "Signal":
        import librosa
        descriptor = torch.from_numpy(
            librosa.feature.spectral_centroid(
                y=self.samples.detach().cpu().numpy()[0][0], 
                sr=self.sample_rate
            )
        ).unsqueeze(0)
        return Signal(
            descriptor, 
            self.stft_frame_rate, 
            self.sample_rate,
            name="librosa_centroid"
        )


    def librosa_bandwidth(self, ) -> "Signal":
        import librosa
        descriptor = torch.from_numpy(
            librosa.feature.spectral_bandwidth(
                y=self.samples.detach().cpu().numpy()[0][0], 
                sr=self.sample_rate
            )
        ).unsqueeze(0)
        return Signal(
            descriptor, 
            self.stft_frame_rate, 
            self.sample_rate,
            name="librosa_bandwidth"
        )

    
    def librosa_flatness(self,) -> "Signal":
        import librosa
        descriptor = torch.from_numpy(
            librosa.feature.spectral_flatness(
                y=self.samples.detach().cpu().numpy()[0][0],
            )
        ).unsqueeze(0)
        return Signal(
            descriptor, 
            self.stft_frame_rate, 
            self.sample_rate,
            name="librosa_flatness"
        )

    

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~ extract all descriptors ~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @property
    def descriptor_fns(self) -> dict:
        return {
        "rms": self.rms,
        "spectral_centroid": self.spectral_centroid,
        "spectral_bandwidth": self.spectral_bandwidth,
        "spectral_flatness": self.spectral_flatness,
        "pitch": self.pitch,
        "periodicity": self.periodicity,
    }


    def extract_descriptors(self, keys: list = None) -> list["Signal"]:
        if keys is None:
            keys = self.descriptor_fns.keys()

        out = []
        for key in keys:
            if key not in self.descriptor_fns:
                raise ValueError(f"Control signal {key} not found!")
            out.append(
                self.descriptor_fns[key]()
            )

        return out


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~ viz helpers!!!!!!!!!!!! ~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def plot(self, 
            ax=None, 
            stem=False, 
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

        if stem:
            ax.stem(data[batch_idx][chan_idx], **kwargs)
        else:
            ax.plot(self.time, data[batch_idx][chan_idx], **kwargs)

        ax.set_title(self.name)
        # plt.tight_layout()

        return fig, ax


    def plot_spec(self, 
        log_magnitude=True,
        ax=None,
    ):
        spec = self.spectrogram()
        if log_magnitude:
            spec = 10 * torch.log10(spec + 1e-10)

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        import librosa
        librosa.display.specshow(
            spec[0][0].detach().cpu().numpy(), 
            sr=self.sample_rate, 
            hop_length=self.stft_params.hop_length, 
            x_axis="time", 
            y_axis="log", 
            ax=ax, 
            cmap="viridis"
        )
        ax.set_xlabel("")

        return fig, ax
    

    def visualize(self,
        rms=None,
        pitch=False,
    ) -> "Signal":
        plt.clf()
        fig, ax = plt.subplots(2, 1, gridspec_kw={"height_ratios": [1, 4]}, sharex=True)

        # add the rms on a twin axis
        if rms is None:
            rms = self.rms()
        else:
            print(f"using provided rms")
        rms.plot(ax=ax[0], stem=False,  color="red")
        ax[0].set_title("")
        # ax[0].set_ylabel("energy (rms)")
        ax[0].legend(["energy (rms)"], loc="upper right")

        # plot the spectrogram
        self.plot_spec(ax=ax[1])

        bw = self.spectral_bandwidth()
        centroid = self.spectral_centroid()

        # overlay the spectral centroid on top, with o markers
        centroid.plot(
            ax=ax[1], stem=False,   
            markersize=3, color="hotpink", label="centroid (brightness)",
        )
        ax[1].set_title("")

        # get the pitch and periodicity
        plot_pitch = False
        if plot_pitch:
            pitch = self.pitch()
            periodicity = self.periodicity()
            # plot the pitch line,
            # set the opacity to whatever the periodicity is
            ax[1].scatter(
                pitch.time, pitch.d[0][0], 
                label="pitch (Hz)", color="blue", alpha=torch.clamp(periodicity.d[0][0], 0.1, 1.0), 
                s=3
            )

        # get the spectral bandwidth
        # plot the centroid ± bandwidth on top, with ---- markers
        ax[1].fill_between(
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

        # add a legend
        ax[1].legend(loc="upper right")

        fig.subplots_adjust(hspace=0.00)
        return fig, ax

    
    @staticmethod
    def toy_signal(duration: float = 5, sample_rate: int = 48000):
        import random
        t = torch.arange(0, duration, 1/sample_rate)
        
        # make a list of 0 - 5 frequencies, their amplitudes, onsets, and offsets, vibratos and tremolos
        events = []
        for i in range(15):
            event = {}
            f0 = random.choice([440, 880])
            event["freq"] = random.choice([f0 * n for n in [1, 0.5, 2, 3, 3/2, 7/8, 5/4]])
            event["onset"] = random.uniform(0, duration /2)
            event["amp"] = random.uniform(0.1, 1)
            event["dur"] = random.uniform(0.1, (duration - event["onset"]) / 8)
            event["vibrato_amt"] = random.uniform(0.1, 0.5)
            event["vibrato_rate"] = random.choice([event['freq'] * n for n in range(1, 5)])
            event["tremolo_amt"] = random.uniform(0.3, 1.0)
            event["tremolo_rate"] = random.uniform(0, 10)
            event["attack"] = random.uniform(0.01, event["dur"]/2)

            # clip the dur to the end of the signal
            if event["onset"] + event["dur"] > duration:
                event["dur"] = duration - event["onset"]
                print(f"clipping event {i} to {event['dur']}")
            events.append(event)


        # make the signal
        samples = torch.zeros_like(t)
        for event in events:
            # make a tensor of samples for this event
            num_samples = int((event["dur"]) * sample_rate)
            
            event_t = torch.arange(0, num_samples, 1) / sample_rate
            # make an envelope
            envelope = torch.ones_like(event_t)
            attack_frames = int(event["attack"] * sample_rate)
            envelope[:attack_frames] = torch.linspace(0, 1, attack_frames)
            envelope[-attack_frames:] = torch.linspace(1, 0, attack_frames)
            # apply the vibrato to the envelope
            vibrato = torch.sin(2 * math.pi * event["vibrato_rate"] * event_t) * event["vibrato_amt"]
            envelope = envelope * (1 + vibrato)
            
            # make per-frequency freq
            osc = torch.sin(
                2 * math.pi * (event["freq"] + torch.sin(2 * math.pi * event["vibrato_rate"] * event_t) * event["vibrato_amt"]) * event_t
            )

            # apply the env to the osc
            osc = osc * envelope

            # place the osc in the right place in the signal
            onset_idx = int(event["onset"] * sample_rate)
            samples[onset_idx:onset_idx + num_samples] += osc[:num_samples] * event["amp"]

        sig = Signal(samples, sample_rate).ensure_max_of_audio()

        return sig



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~ viz helpers!!!!!!!!!!!! ~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def ctrl_plot(sig_dict: dict[str, Signal], ctrls: list[str]):
    plt.close("all")
    n_sigs = len(sig_dict)
    n_ctrls = len(ctrls)

    fig, axes = plt.subplots(figsize=(6 * n_sigs, 4 + 4 * n_ctrls), ncols=n_sigs, nrows=n_ctrls+1, sharex="all", sharey="row") 
    if n_sigs == 1 or n_ctrls == 0:
        axes = axes.reshape(1, -1)
    for i, (name, sig) in enumerate(sig_dict.items()):
        # on the first row, plot the signal's spec
        sig.plot_spec(ax=axes[0, i])
        axes[0, i].set_title(name)
        for j, ctrl in enumerate(ctrls):
            # on the subsequent rows, plot the control signal
            ctrl_sig: Signal  = sig.descriptor_fns[ctrl]()
            ctrl_sig.plot(ax=axes[j+1, i], )
            axes[j+1, i].set_title(f"{ctrl}")

    fig.tight_layout()
    return fig

if __name__ == "__main__":
    # make a toy signal
    sig = Signal.toy_signal()

    sig.visualize(rms=sig.rms())
    img0 = plt2pil()

    # plot the signal
    rms = sig.rms().low_pass_decimate(10)
    fig, ax = sig.visualize(rms=rms)
    img1 = plt2pil()
    
    rms = sig.rms().low_pass(10) 
    fig, ax = sig.visualize(rms=rms)
    img2 = plt2pil()

    vertical_pil_stack([img0, img1, img2]).save("test.png")
    sig.write("test.wav")
    


    