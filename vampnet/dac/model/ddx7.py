import torch
import torch.nn as nn
import torch.fft as fft
import numpy as np
# import librosa as li
import math

_DB_RANGE = 80.0 #Min loudness
_REF_DB = 20.7  # White noise, amplitude=1.0, n_fft=2048
_F0_RANGE = 127

def safe_log(x,eps=1e-7):
    eps = torch.tensor(eps)
    return torch.log(x + eps)

def safe_divide(numerator, denominator, eps=1e-7):
    """Avoid dividing by zero by adding a small epsilon."""
    eps = torch.tensor(eps)
    safe_denominator = torch.where(denominator == 0.0, eps, denominator)
    return numerator / safe_denominator

def logb(x, base=2.0, eps=1e-5):
    """Logarithm with base as an argument."""
    return safe_divide(safe_log(x, eps), safe_log(base, eps), eps)

def hz_to_midi(frequencies):
    """Torch-compatible hz_to_midi function."""
    notes = 12.0 * (logb(frequencies, 2.0) - logb(440.0, 2.0)) + 69.0
    notes = torch.where(torch.le(frequencies, torch.zeros(1).to(frequencies)),
                        torch.zeros(1).to(frequencies), notes)
    return notes


@torch.no_grad()
def cumsum_nd(in_tensor,wrap_value=None):
    '''
    cumsum_nd() : cummulative sum - non differentiable and with wrap value.

    The problem with cumsum: when we work with phase tensors that are too large
    (i.e. more than a few tenths of seconds) cumsum gets to accumulate steps
    over a very large window, and it seems the float point variable loses precision.

    This workaround computes the accumulation step by step, resetting the
    accumulator in order for it to avoid to lose precision.

    NOTE: This implementation is very slow, and can't be used during training,
    only for final audio rendering on the test set.

    Assumes a tensor format used for audio rendering. [batch,len,1]

    NOTE:  Non integer frequency ratios do not work using current synthesis approach,
    because we render a common phase (wrapped using cumsum_nd) and then we multiply it
    by the frequency ratio. This introduces a misalignment if we multiply the wrapped phase
    by a non-integer frequency ratio.

    TODO: implement an efficient vectorial cumsum with wrapping we can use to accumulate
          phases from all oscillators separately
    '''
    print("[WARNING] Using non differentiable cumsum. Non-integer frequency ratios wont render well.")
    input_len = in_tensor.size()[1]
    nb = in_tensor.size()[0]
    acc = torch.zeros([nb,1,1])
    out_tensor = torch.zeros([nb,input_len,1])
    #print("in size{} - out size{}".format(in_tensor.size(),out_tensor.size()))
    for i in range(input_len):
        acc += in_tensor[:,i,0]
        if(wrap_value is not None):
            acc = acc - (acc > wrap_value)*wrap_value
        out_tensor[:,i,0] = acc
    return out_tensor



@torch.no_grad()
def mean_std_loudness(dataset):
    mean = 0
    std = 0
    n = 0
    for _, _, l in dataset:
        n += 1
        mean += (l.mean().item() - mean) / n
        std += (l.std().item() - std) / n
    return mean, std


def multiscale_fft(signal, scales, overlap):
    stfts = []
    for s in scales:
        S = torch.stft(
            signal,
            s,
            int(s * (1 - overlap)),
            s,
            torch.hann_window(s).to(signal),
            True,
            normalized=True,
            return_complex=True,
        ).abs()
        stfts.append(S)
    return stfts


def resample(x, factor: int):
    batch, frame, channel = x.shape
    x = x.permute(0, 2, 1).reshape(batch * channel, 1, frame)

    window = torch.hann_window(
        factor * 2,
        dtype=x.dtype,
        device=x.device,
    ).reshape(1, 1, -1)
    y = torch.zeros(x.shape[0], x.shape[1], factor * x.shape[2]).to(x)
    y[..., ::factor] = x
    y[..., -1:] = x[..., -1:]
    y = torch.nn.functional.pad(y, [factor, factor])
    y = torch.nn.functional.conv1d(y, window)[..., :-1]

    y = y.reshape(batch, channel, factor * frame).permute(0, 2, 1)

    return y


def upsample(signal, factor,mode='nearest'):
    signal = signal.permute(0, 2, 1)
    signal = nn.functional.interpolate(signal, size=signal.shape[-1] * factor,mode=mode)
    return signal.permute(0, 2, 1)


# def extract_loudness(signal, sampling_rate, block_size, n_fft=2048):
#     S = li.stft(
#         signal,
#         n_fft=n_fft,
#         hop_length=block_size,
#         win_length=n_fft,
#         center=True,
#     )
#     S = np.log(abs(S) + 1e-7)
#     f = li.fft_frequencies(sampling_rate, n_fft)
#     a_weight = li.A_weighting(f)

#     S = S + a_weight.reshape(-1, 1)

#     S = np.mean(S, 0)[..., :-1]

#     return S



def get_mlp(in_size, hidden_size, n_layers):
    channels = [in_size] + (n_layers) * [hidden_size]
    net = []
    for i in range(n_layers):
        net.append(nn.Linear(channels[i], channels[i + 1]))
        net.append(nn.LayerNorm(channels[i + 1]))
        net.append(nn.LeakyReLU())
    return nn.Sequential(*net)


def get_gru(n_input, hidden_size, num_layers=1):
    return nn.GRU(n_input * hidden_size, hidden_size, num_layers=num_layers, batch_first=True)


def amp_to_impulse_response(amp, target_size):
    amp = torch.stack([amp, torch.zeros_like(amp)], -1)
    amp = torch.view_as_complex(amp)
    amp = fft.irfft(amp)

    filter_size = amp.shape[-1]

    amp = torch.roll(amp, filter_size // 2, -1)
    win = torch.hann_window(filter_size, dtype=amp.dtype, device=amp.device)

    amp = amp * win

    amp = nn.functional.pad(amp, (0, int(target_size) - int(filter_size)))
    amp = torch.roll(amp, -filter_size // 2, -1)

    return amp


def fft_convolve(signal, kernel):
    signal = nn.functional.pad(signal, (0, signal.shape[-1]))
    kernel = nn.functional.pad(kernel, (kernel.shape[-1], 0))

    output = fft.irfft(fft.rfft(signal) * fft.rfft(kernel))
    output = output[..., output.shape[-1] // 2:]

    return output


def harmonic_synth(pitch, amplitudes, sampling_rate,use_safe_cumsum=False):

    if(use_safe_cumsum==True):
        omega = cumsum_nd(2 * np.pi * pitch / sampling_rate, 2*np.pi)
    else:
        omega = torch.cumsum(2 * np.pi * pitch / sampling_rate, 1)

    n_harmonic = amplitudes.shape[-1]
    omegas = omega * torch.arange(1, n_harmonic + 1).to(omega)
    signal = (torch.sin(omegas) * amplitudes).sum(-1, keepdim=True)
    return signal

OP6=5
OP5=4
OP4=3
OP3=2
OP2=1
OP1=0

def fm_2stack2(pitch, ol, fr, sampling_rate,max_ol,use_safe_cumsum=False):

    if(use_safe_cumsum==True):
        omega = cumsum_nd(2 * np.pi * pitch / sampling_rate, 2*np.pi)
    else:
        omega = torch.cumsum(2 * np.pi * pitch / sampling_rate, 1)

    # Torch unsqueeze with dim -1 adds a new dimension at the end of ol to match phases.

    op4_phase =  fr[OP4] * omega
    op4_output = torch.unsqueeze(ol[:,:,OP4], dim=-1) * torch.sin(op4_phase)

    op3_phase =  fr[OP3] * omega + 2 * np.pi * op4_output
    op3_output = torch.unsqueeze(ol[:,:,OP3], dim=-1) * torch.sin(op3_phase) # output of stack of 2

    op2_phase =  fr[OP2] * omega
    op2_output = torch.unsqueeze(ol[:,:,OP2], dim=-1) * torch.sin(op2_phase)

    op1_phase =  fr[OP1] * omega + 2 * np.pi * op2_output
    op1_output = torch.unsqueeze(ol[:,:,OP1], dim=-1) * torch.sin(op1_phase) # output of stack of 2

    return (op3_output + op1_output)/max_ol

def fm_1stack2(pitch, ol, fr, sampling_rate,max_ol,use_safe_cumsum=False):

    if(use_safe_cumsum==True):
        omega = cumsum_nd(2 * np.pi * pitch / sampling_rate, 2*np.pi)
    else:
        omega = torch.cumsum(2 * np.pi * pitch / sampling_rate, 1)

    # Torch unsqueeze with dim -1 adds a new dimension at the end of ol to match phases.

    op2_phase =  fr[OP2] * omega
    op2_output = torch.unsqueeze(ol[:,:,OP2], dim=-1) * torch.sin(op2_phase)

    op1_phase =  fr[OP1] * omega + 2 * np.pi * op2_output
    op1_output = torch.unsqueeze(ol[:,:,OP1], dim=-1) * torch.sin(op1_phase) # output of stack of 2

    return op1_output/max_ol


def fm_1stack4(pitch, ol, fr, sampling_rate,max_ol,use_safe_cumsum=False):

    if(use_safe_cumsum==True):
        omega = cumsum_nd(2 * np.pi * pitch / sampling_rate, 2*np.pi)
    else:
        omega = torch.cumsum(2 * np.pi * pitch / sampling_rate, 1)

    # Torch unsqueeze with dim -1 adds a new dimension at the end of ol to match phases.

    op4_phase =  fr[OP4] * omega
    op4_output = torch.unsqueeze(ol[:,:,OP4], dim=-1) * torch.sin(op4_phase)

    op3_phase =  fr[OP3] * omega + 2 * np.pi * op4_output
    op3_output = torch.unsqueeze(ol[:,:,OP3], dim=-1) * torch.sin(op3_phase) # output of stack of 4

    op2_phase =  fr[OP2] * omega + 2 * np.pi * op3_output
    op2_output = torch.unsqueeze(ol[:,:,OP2], dim=-1) * torch.sin(op2_phase)

    op1_phase =  fr[OP1] * omega + 2 * np.pi * op2_output
    op1_output = torch.unsqueeze(ol[:,:,OP1], dim=-1) * torch.sin(op1_phase) # output of stack of 2

    return op1_output/max_ol


'''
Ablated Brass FM Synth - with phase wrapping (it does not change behaviour)
     OP4->OP3->|
          OP2->|->OP1->out

'''
def fm_ablbrass_synth(pitch, ol, fr, sampling_rate,max_ol,use_safe_cumsum=False):

    if(use_safe_cumsum==True):
        omega = cumsum_nd(2 * np.pi * pitch / sampling_rate, 2*np.pi)
    else:
        omega = torch.cumsum(2 * np.pi * pitch / sampling_rate, 1)

    # Torch unsqueeze with dim -1 adds a new dimension at the end of ol to match phases.

    op4_phase =  fr[OP4] * omega
    op4_output = torch.unsqueeze(ol[:,:,OP4], dim=-1) * torch.sin(op4_phase % (2*np.pi))

    op3_phase =  fr[OP3] * omega + 2 * np.pi * op4_output
    op3_output = torch.unsqueeze(ol[:,:,OP3], dim=-1) * torch.sin(op3_phase % (2*np.pi)) # output of stack of 2

    op2_phase =  fr[OP2] * omega
    op2_output = torch.unsqueeze(ol[:,:,OP2], dim=-1) * torch.sin(op2_phase % (2*np.pi)) # output stack of 1

    op1_phase =  fr[OP1] * omega + 2 * np.pi * (op2_output + op3_output)
    op1_output = torch.unsqueeze(ol[:,:,OP1], dim=-1) * torch.sin(op1_phase % (2*np.pi)) # global carrier

    return op1_output/max_ol

'''
String FM Synth - with phase wrapping (it does not change behaviour)
PATCH NAME: STRINGS 1
OP6->OP5->OP4->OP3 |
       (R)OP2->OP1 |->out
'''
def fm_string_synth(pitch, ol, fr, sampling_rate,max_ol,use_safe_cumsum=False):

    if(use_safe_cumsum==True):
        omega = cumsum_nd(2 * np.pi * pitch / sampling_rate, 2*np.pi)
    else:
        omega = torch.cumsum(2 * np.pi * pitch / sampling_rate, 1)

    # Torch unsqueeze with dim -1 adds a new dimension at the end of ol to match phases.
    op6_phase =  fr[OP6] * omega
    op6_output = torch.unsqueeze(ol[:,:,OP6], dim=-1) * torch.sin(op6_phase % (2*np.pi))

    op5_phase =  fr[OP5] * omega + 2 * np.pi * op6_output
    op5_output = torch.unsqueeze(ol[:,:,OP5], dim=-1)*torch.sin(op5_phase % (2*np.pi))

    op4_phase =  fr[OP4] * omega + 2 * np.pi * op5_output
    op4_output = torch.unsqueeze(ol[:,:,OP4], dim=-1) * torch.sin(op4_phase % (2*np.pi))

    op3_phase =  fr[OP3] * omega + 2 * np.pi * op4_output
    op3_output = torch.unsqueeze(ol[:,:,OP3], dim=-1) * torch.sin(op3_phase % (2*np.pi)) # output of stack of 4

    op2_phase =  fr[OP2] * omega
    op2_output = torch.unsqueeze(ol[:,:,OP2], dim=-1) * torch.sin(op2_phase % (2*np.pi))

    op1_phase =  fr[OP1] * omega + 2 * np.pi * op2_output
    op1_output = torch.unsqueeze(ol[:,:,OP1], dim=-1) * torch.sin(op1_phase % (2*np.pi)) # output of stack of 2

    return (op3_output + op1_output)/max_ol

'''
Flute FM Synth - with phase wrapping (it does not change behaviour)
PATCH NAME: FLUTE 1
(R)OP6->OP5->|
   OP4->OP3->|
        OP2->|->OP1->out
'''
def fm_flute_synth(pitch, ol, fr, sampling_rate,max_ol,use_safe_cumsum=False):

    if(use_safe_cumsum==True):
        omega = cumsum_nd(2 * np.pi * pitch / sampling_rate, 2*np.pi)
    else:
        omega = torch.cumsum(2 * np.pi * pitch / sampling_rate, 1)

    # Torch unsqueeze with dim -1 adds a new dimension at the end of ol to match phases.
    op6_phase =  fr[OP6] * omega
    op6_output = torch.unsqueeze(ol[:,:,OP6], dim=-1) * torch.sin(op6_phase % (2*np.pi))

    op5_phase =  fr[OP5] * omega + 2 * np.pi * op6_output
    op5_output = torch.unsqueeze(ol[:,:,OP5], dim=-1)*torch.sin(op5_phase % (2*np.pi)) # output of stack of 2

    op4_phase =  fr[OP4] * omega
    op4_output = torch.unsqueeze(ol[:,:,OP4], dim=-1) * torch.sin(op4_phase % (2*np.pi))

    op3_phase =  fr[OP3] * omega + 2 * np.pi * op4_output
    op3_output = torch.unsqueeze(ol[:,:,OP3], dim=-1) * torch.sin(op3_phase % (2*np.pi)) # output of stack of 2

    op2_phase =  fr[OP2] * omega
    op2_output = torch.unsqueeze(ol[:,:,OP2], dim=-1) * torch.sin(op2_phase % (2*np.pi)) # output stack of 1

    op1_phase =  fr[OP1] * omega + 2 * np.pi * (op2_output + op3_output + op5_output)
    op1_output = torch.unsqueeze(ol[:,:,OP1], dim=-1) * torch.sin(op1_phase % (2*np.pi)) # carrier

    return op1_output/max_ol

'''
Brass FM Synth - with phase wrapping (it does not change behaviour)
PATCH NAME: BRASS 3
OP6->OP5->OP4->|
       (R)OP3->|
          OP2->|->OP1->out
'''
def fm_brass_synth(pitch, ol, fr, sampling_rate,max_ol,use_safe_cumsum=False):

    if(use_safe_cumsum==True):
        omega = cumsum_nd(2 * np.pi * pitch / sampling_rate, 2*np.pi)
    else:
        omega = torch.cumsum(2 * np.pi * pitch / sampling_rate, 1)

    # Torch unsqueeze with dim -1 adds a new dimension at the end of ol to match phases.
    op6_phase =  fr[OP6] * omega
    op6_output = torch.unsqueeze(ol[:,:,OP6], dim=-1) * torch.sin(op6_phase % (2*np.pi))

    op5_phase =  fr[OP5] * omega + 2 * np.pi * op6_output
    op5_output = torch.unsqueeze(ol[:,:,OP5], dim=-1)*torch.sin(op5_phase % (2*np.pi))

    op4_phase =  fr[OP4] * omega + 2 * np.pi * op5_output
    op4_output = torch.unsqueeze(ol[:,:,OP4], dim=-1) * torch.sin(op4_phase % (2*np.pi)) # output of stack of 3

    op3_phase =  fr[OP3] * omega
    op3_output = torch.unsqueeze(ol[:,:,OP3], dim=-1) * torch.sin(op3_phase % (2*np.pi)) # output of stack of 1

    op2_phase =  fr[OP2] * omega
    op2_output = torch.unsqueeze(ol[:,:,OP2], dim=-1) * torch.sin(op2_phase % (2*np.pi)) # output stack of 1

    op1_phase =  fr[OP1] * omega + 2 * np.pi * (op2_output + op3_output + op4_output)
    op1_output = torch.unsqueeze(ol[:,:,OP1], dim=-1) * torch.sin(op1_phase % (2*np.pi)) # carrier

    return op1_output/max_ol