from typing import Union, List

import torch
import torchaudio.transforms as T
import numpy as np
from audiotools import AudioSignal
import random

from msclap import CLAP

# setting sample rate
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else "cpu" #

class AbstractCLAPWrapper:
    def preprocess_audio(self, signal: AudioSignal):
        raise NotImplementedError("implement me :)")
    
    def get_audio_embeddings(self, signal: AudioSignal):
        raise NotImplementedError()
    
    def get_text_embeddings(self, text: Union[str, List[str]]):
        raise NotImplementedError()
    

""" utility wrapper around MS CLAP model! """
class MSCLAPWrapper(AbstractCLAPWrapper):

    def __init__(self):
        self.clap_model = CLAP(version = '2023', use_cuda=True)

    #testing just the clap_model.load_audio() !!
    def resample(self, signal: AudioSignal, resample=True):
        """
        trying to see if resampling step in read_audio (step 1) is the issue, but it seems its from the very beginning w/ reading in the file
        """
        audio_time_series = signal.samples
        resample_rate = self.clap_model.args.sampling_rate
        sample_rate = signal.sample_rate #check

        if resample and resample_rate != sample_rate:
            resampler = T.Resample(sample_rate, resample_rate)
            resampler.to(signal.device)
            audio_time_series = resampler(audio_time_series)
            signal.samples = audio_time_series
            signal.sample_rate = resample_rate
            
        return signal

    def audio_trim(self, audio_time_series, audio_duration, sample_rate):
        audio_time_series = audio_time_series.squeeze(0).squeeze(0)

        #if audio duration is shorter than 7 seconds, repeat samples
        if audio_duration*sample_rate >= audio_time_series.shape[0]: 
            repeat_factor = int(np.ceil((audio_duration*sample_rate) /
                                        audio_time_series.shape[0]))
            # Repeat audio_time_series by repeat_factor to match audio_duration
            audio_time_series = audio_time_series.repeat(repeat_factor)
            # remove excess part of audio_time_series
            audio_time_series = audio_time_series[0:audio_duration*sample_rate]
        else:
        # audio_time_series is longer than predefined audio duration (7s),
        # so audio_time_series is trimmed
            start_index = random.randrange(
                audio_time_series.shape[0] - audio_duration*sample_rate)
            audio_time_series = audio_time_series[start_index:start_index +
                                                audio_duration*sample_rate]
        return audio_time_series.unsqueeze(0).unsqueeze(0).float()

    def preprocess_audio(self, signal: AudioSignal):
        sig_resamp = []
        for i in range(signal.shape[0]):
            _sig = self.resample(signal[i]) #uses CLAP function on AudioSignal.samples
            _sig.to_mono()
            _sig.samples = self.audio_trim(
                _sig.samples, 
                self.clap_model.args.duration, 
                self.clap_model.args.sampling_rate
            ) #should work for batches, NOTE: might be good to vectorize
            sig_resamp.append(_sig)

        return AudioSignal.batch(sig_resamp)

    #get_audio_embeddings from preprocessed AudioSignal (resampling, duration reworking)
    def get_audio_embed(self, preprocessed_audio: AudioSignal): 
        preprocessed_audio = preprocessed_audio.reshape(preprocessed_audio.shape[0], preprocessed_audio.shape[2])
        return self.clap_model.clap.audio_encoder(preprocessed_audio)[0]

    #preprocess raw AudioSignal + get_audio_embeddings from that
    # NOTE: SHOULD THIS BE RENAMED AS self.get_audio_embeddings()
    def get_audio_embeddings(self, signal: AudioSignal):
        return self.get_audio_embed(self.preprocess_audio(signal).samples)

    def get_text_embeddings(self, texts):
        return self.clap_model.get_text_embeddings(texts)
    
