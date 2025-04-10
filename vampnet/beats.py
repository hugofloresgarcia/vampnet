import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import List
from typing import Tuple
from typing import Union

import librosa
import torch
import numpy as np
from audiotools import AudioSignal


logging.basicConfig(level=logging.INFO)

###################
# beat sync utils #
###################

AGGREGATOR_REGISTRY = {
    "mean": np.mean,
    "median": np.median,
    "max": np.max,
    "min": np.min,
}


def list_aggregators() -> list:
    return list(AGGREGATOR_REGISTRY.keys())


@dataclass
class TimeSegment:
    start: float
    end: float

    @property
    def duration(self):
        return self.end - self.start

    def __str__(self) -> str:
        return f"{self.start} - {self.end}"

    def find_overlapping_segment(
        self, segments: List["TimeSegment"]
    ) -> Union["TimeSegment", None]:
        """Find the first segment that overlaps with this segment, or None if no segment overlaps"""
        for s in segments:
            if s.start <= self.start and s.end >= self.end:
                return s
        return None


def mkdir(path: Union[Path, str]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p



###################
#    beat data    #
###################
@dataclass
class BeatSegment(TimeSegment):
    downbeat: bool = False  # if there's a downbeat on the start_time


class Beats:
    def __init__(self, beat_times, downbeat_times):
        if isinstance(beat_times, np.ndarray):
            beat_times = beat_times.tolist()
        if isinstance(downbeat_times, np.ndarray):
            downbeat_times = downbeat_times.tolist()
        self._beat_times = beat_times
        self._downbeat_times = downbeat_times
        self._use_downbeats = False

    def use_downbeats(self, use_downbeats: bool = True):
        """use downbeats instead of beats when calling beat_times"""
        self._use_downbeats = use_downbeats

    def beat_segments(self, signal: AudioSignal) -> List[BeatSegment]:
        """
        segments a song into time segments corresponding to beats.
        the first segment starts at 0 and ends at the first beat time.
        the last segment starts at the last beat time and ends at the end of the song.
        """
        beat_times = self._beat_times.copy()
        downbeat_times = self._downbeat_times
        beat_times.insert(0, 0)
        beat_times.append(signal.signal_duration)

        downbeat_ids = np.intersect1d(beat_times, downbeat_times, return_indices=True)[
            1
        ]
        is_downbeat = [
            True if i in downbeat_ids else False for i in range(len(beat_times))
        ]
        segments = [
            BeatSegment(start_time, end_time, downbeat)
            for start_time, end_time, downbeat in zip(
                beat_times[:-1], beat_times[1:], is_downbeat
            )
        ]
        return segments

    def get_beats(self) -> np.ndarray:
        """returns an array of beat times, in seconds
        if downbeats is True, returns an array of downbeat times, in seconds
        """
        return np.array(
            self._downbeat_times if self._use_downbeats else self._beat_times
        )

    @property
    def beat_times(self) -> np.ndarray:
        """return beat times"""
        return np.array(self._beat_times)

    @property
    def downbeat_times(self) -> np.ndarray:
        """return downbeat times"""
        return np.array(self._downbeat_times)

    def beat_times_to_feature_frames(
        self, signal: AudioSignal, features: np.ndarray
    ) -> np.ndarray:
        """convert beat times to frames, given an array of time-varying features"""
        beat_times = self.get_beats()
        beat_frames = (
            beat_times * signal.sample_rate / signal.signal_length * features.shape[-1]
        ).astype(np.int64)
        return beat_frames

    def sync_features(
        self, feature_frames: np.ndarray, features: np.ndarray, aggregate="median"
    ) -> np.ndarray:
        """sync features to beats"""
        if aggregate not in AGGREGATOR_REGISTRY:
            raise ValueError(f"unknown aggregation method {aggregate}")

        return librosa.util.sync(
            features, feature_frames, aggregate=AGGREGATOR_REGISTRY[aggregate]
        )

    def to_json(self) -> dict:
        """return beats and downbeats as json"""
        return {
            "beats": self._beat_times,
            "downbeats": self._downbeat_times,
            "use_downbeats": self._use_downbeats,
        }

    @classmethod
    def from_dict(cls, data: dict):
        """load beats and downbeats from json"""
        inst = cls(data["beats"], data["downbeats"])
        inst.use_downbeats(data["use_downbeats"])
        return inst

    def save(self, output_dir: Path):
        """save beats and downbeats to json"""
        mkdir(output_dir)
        with open(output_dir / "beats.json", "w") as f:
            json.dump(self.to_json(), f)

    @classmethod
    def load(cls, input_dir: Path):
        """load beats and downbeats from json"""
        beats_file = Path(input_dir) / "beats.json"
        with open(beats_file, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


###################
#  beat tracking  #
###################


class BeatTracker:
    def extract_beats(self, signal: AudioSignal) -> Tuple[np.ndarray, np.ndarray]:
        """extract beats from an audio signal"""
        raise NotImplementedError

    def __call__(self, signal: AudioSignal) -> Beats:
        """extract beats from an audio signal
        NOTE: if the first beat (and/or downbeat) is detected within the first 100ms of the audio,
        it is discarded. This is to avoid empty bins with no beat synced features in the first beat.
        Args:
            signal (AudioSignal): signal to beat track
        Returns:
            Tuple[np.ndarray, np.ndarray]: beats and downbeats
        """
        beats, downbeats = self.extract_beats(signal)
        return Beats(beats, downbeats)


class WaveBeat(BeatTracker):
    def __init__(self, ckpt_path: str = "checkpoints/wavebeat", device: str = "cpu"):
        from wavebeat.dstcn import dsTCNModel

        model = dsTCNModel.load_from_checkpoint(ckpt_path, map_location=torch.device(device))
        model.eval()

        self.device = device
        self.model = model

    def extract_beats(self, signal: AudioSignal) -> Tuple[np.ndarray, np.ndarray]:
        """returns beat and downbeat times, in  seconds"""
        # extract beats
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
        beats, downbeats = self.model.predict_beats_from_array(
            audio=signal.audio_data.squeeze(0),
            sr=signal.sample_rate,
            use_gpu=torch.cuda.is_available(),
        )

        return beats, downbeats


class MadmomBeats(BeatTracker):
    def __init__(self):
        raise NotImplementedError

    def extract_beats(self, signal: AudioSignal) -> Tuple[np.ndarray, np.ndarray]:
        """returns beat and downbeat times, in  seconds"""
        pass


BEAT_TRACKER_REGISTRY = {
    "wavebeat": WaveBeat,
    "madmom": MadmomBeats,
}


def list_beat_trackers() -> list:
    return list(BEAT_TRACKER_REGISTRY.keys())


def load_beat_tracker(beat_tracker: str, **kwargs) -> BeatTracker:
    if beat_tracker not in BEAT_TRACKER_REGISTRY:
        raise ValueError(
            f"Unknown beat tracker {beat_tracker}. Available: {list_beat_trackers()}"
        )

    return BEAT_TRACKER_REGISTRY[beat_tracker](**kwargs)