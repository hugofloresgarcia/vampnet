import random
import vampnet
import vampnet.signal as sn
from vampnet.interface import Interface
import audiotools as at
import torch

codec_ckpt = "/home/hugo/.cache/descript/dac/weights_44khz_8kbps_0.0.1.pth"

codec = vampnet.dac.DAC.load(codec_ckpt)
device = "cuda" if torch.cuda.is_available() else "cpu"

at.util.seed(0)

# load the dataset for playing w/
# from scripts.exp.train import build_datasets

# train_data, val_data = build_datasets(
#     sample_rate=codec.sample_rate, 
#     db_path="scratch/data-fast/sm.db", 
#     query="SELECT * FROM audio_file JOIN dataset where dataset.name = 'vctk'"
# )


# load an audio file
sig = sn.read_from_file("assets/example.wav")
# sig = val_data[0]["sig"]

# sig.wav = torch.cat([sig.wav, sig.wav, sig.wav], dim=-1)

# cut sig to hop length
sig = sn.trim_to_s(sig, 5.0)
sig.wav = sn.cut_to_hop_length(sig.wav, codec.hop_length)
sig = sn.normalize(sig, -16)
sig = sn.resample(sig, 44100)


codec.to(device)
sig = sig.to(device)

wav = sig.wav
nt = wav.shape[-1]
wav = codec.preprocess(wav, 44100)
codes = codec.encode(wav)["codes"]

z = codec.quantizer.from_codes(codes)[0]

# decode
wav = codec.decode(z)
outsig = sn.Signal(wav, sig.sr)
print(wav.shape)

# write to file
sn.write(outsig, "scratch/dac-recons.wav")
sn.write(sig, "scratch/dac-orig.wav")
