import random
import vampnet
import vampnet.signal as sn
from vampnet.interface import EmbeddedInterface
import audiotools as at

codec = vampnet.dac.DAC.load("/home/hugo/.cache/descript/dac/weights_44khz_8kbps_0.0.1.pth")
eiface = EmbeddedInterface(
    codec=codec,
    coarse=vampnet.VampNet(),
    chunk_size_s=10
)

# load an audio file
sig = sn.read_from_file("assets/example.wav")
# cut sig to hop length
sig.wav = sn.cut_to_hop_length(sig.wav, eiface.codec.hop_length)

sig = sig.to("cuda")
codec.to("cuda")

codes = eiface.encode(sig.wav)
print(codes.shape)

# decode
wav = eiface.decode(codes)
outsig = sn.Signal(wav, sig.sr)
print(wav.shape)

# write to file
sn.write(outsig, "scratch/ex-recons.wav")