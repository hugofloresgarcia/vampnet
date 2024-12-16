from .ddx7 import *
from .dac import DAC

def exp_sigmoid(x):
    return 2 * torch.sigmoid(x)**(math.log(10)) + 1e-7

def remove_above_nyquist(amplitudes, pitch, sampling_rate):
    n_harm = amplitudes.shape[-1]
    pitches = pitch * torch.arange(1, n_harm + 1).to(pitch)
    aa = (pitches < sampling_rate / 2).float() + 1e-4
    return amplitudes * aa

class FMSynth(nn.Module):
    def __init__(self,sample_rate,block_size,fr=[1,1,1,1,3,14],max_ol=2,
        scale_fn = torch.sigmoid,synth_module='fmstrings'):
        super().__init__()
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.reverb = Reverb(length=sample_rate, sample_rate=sample_rate)
        fr = torch.tensor(fr) # Frequency Ratio
        self.register_buffer("fr", fr) #Non learnable but sent to GPU if declared as buffers, and stored in model dictionary
        self.scale_fn = scale_fn
        self.use_cumsum_nd = False
        self.max_ol = max_ol

        available_synths = {
            'fmbrass': fm_brass_synth,
            'fmflute': fm_flute_synth,
            'fmstrings': fm_string_synth,
            'fmablbrass': fm_ablbrass_synth,
            '2stack2': fm_2stack2,
            '1stack2':fm_1stack2,
            '1stack4': fm_1stack4}

        self.synth_module = available_synths[synth_module]

    def forward(self,controls):

        ol = self.max_ol*self.scale_fn(controls['ol'])
        ol_up = upsample(ol, self.block_size,'linear')
        f0_up = upsample(controls['f0_hz'], self.block_size,'linear')
        signal = self.synth_module(f0_up,
                                ol_up,
                                self.fr,
                                self.sample_rate,
                                self.max_ol,
                                self.use_cumsum_nd)
        #reverb part
        signal = self.reverb(signal)

        synth_out = {
            'synth_audio': signal,
            'ol': ol,
            'f0_hz': controls['f0_hz']
            }
        return synth_out

class HNSynth(nn.Module):
    def __init__(self,
        sample_rate,
        block_size,
        scale_fn = exp_sigmoid
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.reverb = Reverb(length=sample_rate, sample_rate=sample_rate)
        self.use_cumsum_nd = False
        self.scale_fn = scale_fn

    # expects: f0, harmonic_distribution, amplitude, noise_bands
    def forward(self, controls: dict):
        """
        the controls dict must have the following keys: 
        - f0_hz: fundamental frequency in Hz (shape [seq, batch, 1])
        - harmonic_distribution: harmonic distribution (shape [seq, batch, 100])
        - amplitude: amplitude (shape [seq, batch, 1])
        - noise_bands: noise bands (shape [seq, batch, 65])
        """

        harmonics = self.scale_fn(controls['harmonic_distribution'])
        noise_bands = self.scale_fn(controls['noise_bands'])
        total_amp = self.scale_fn(controls['amplitude'])

        print("~~~~~~")
        print(f"f0_hz")
        # print(controls['f0_hz'])
        print(controls['f0_hz'].min().item(), controls['f0_hz'].max().item())
        print(f"noise bands:")
        print(controls["noise_bands"].min().item(), controls["noise_bands"].max().item())
        print()

        harmonics = remove_above_nyquist(
            harmonics,
            controls['f0_hz'],
            self.sample_rate,
        )
        harmonics /= harmonics.sum(-1, keepdim=True)
        harmonics *= total_amp

        harmonics_up = upsample(harmonics, self.block_size)
        f0_up = upsample(controls['f0_hz'], self.block_size,'linear')

        harmonic = harmonic_synth(f0_up, harmonics_up, self.sample_rate, self.use_cumsum_nd)
        impulse = amp_to_impulse_response(noise_bands, self.block_size)

        noise = torch.rand(
            impulse.shape[0],
            impulse.shape[1],
            self.block_size,
            ).to(impulse) * 2 - 1

        noise = fft_convolve(noise, impulse).contiguous()
        noise = noise.reshape(noise.shape[0], -1, 1)

        signal = harmonic + noise

        #reverb part
        signal = self.reverb(signal)
        synth_out = {
            'synth_audio': signal,
            'harmonic_distribution': harmonics,
            'noise_bands': noise_bands,
            'f0_hz': controls['f0_hz']
        }

        return synth_out

class Reverb(nn.Module):
    def __init__(self, length, sample_rate, initial_wet=0, initial_decay=5):
        super().__init__()
        self.length = length
        self.sample_rate = sample_rate

        self.noise = nn.Parameter((torch.rand(length) * 2 - 1).unsqueeze(-1))
        self.decay = nn.Parameter(torch.tensor(float(initial_decay)))
        self.wet = nn.Parameter(torch.tensor(float(initial_wet)))

        t = torch.arange(self.length) / self.sample_rate
        t = t.reshape(1, -1, 1)
        self.register_buffer("t", t)

    def build_impulse(self):
        t = torch.exp(-nn.functional.softplus(-self.decay) * self.t * 500)
        noise = self.noise * t
        impulse = noise * torch.sigmoid(self.wet)
        impulse[:, 0] = 1
        return impulse

    def forward(self, x):
        lenx = x.shape[1]
        impulse = self.build_impulse()
        impulse = nn.functional.pad(impulse, (0, 0, 0, lenx - self.length))

        x = fft_convolve(x.squeeze(-1), impulse.squeeze(-1)).unsqueeze(-1)

        return x


class DDSP_Decoder(nn.Module):
    def __init__(self,decoder,synth):
        super().__init__()
        net = []
        net.append(decoder)
        net.append(synth)
        self.net = nn.Sequential(*net)

    def forward(self,x):
        return self.net(x)

    def get_sr(self):
        return self.net[1].sample_rate

    def enable_cumsum_nd(self):
        self.net[1].use_cumsum_nd=True

    def get_params(self,param):
        if(param == 'reverb_decay'):
            return self.net[1].reverb.decay.item()
        if(param == 'reverb_wet'):
            return self.net[1].reverb.wet.item()

# map from linear-midi to Hz, make sure range is 20-2kHz
class FreqMap(nn.Module):
    def forward(self, x):
        # restrict range to 20-2kHz
        # 20Hz to midi is 15.4868
        # 2kHz to midi is 95.2131
        x = torch.sigmoid(x) 
        # map to Hz (linear)
        x = 20 + (1000 - 20) * x

        # map to Hz (log)
        # x = 15.4868 + (95.2131 - 15.4868) * x
        # map to Hz
        # x = 2**((x-69)/12) * 440
        return x

'''
GRU-Based decoder for HpN Baseline
'''
class RnnFCDecoder(nn.Module):
    def __init__(self, hidden_size=512, sample_rate=16000,
                 input_keys=None,
                 input_sizes=[1,1,16],
                 output_keys=['amplitude','f0', 'harmonic_distribution','noise_bands'],
                 output_sizes=[1,1,100,65], 
                 num_mlp_layers=3,
                 num_gru_layers=1):
        super().__init__()
        self.input_keys = input_keys
        self.input_sizes = input_sizes
        n_keys = len(input_keys)
        # Generate MLPs of size: in_size: 1 ; n_layers = 3 (with layer normalization and leaky relu)
        if(n_keys == 1):
            self.in_mlps = nn.ModuleList([get_mlp(input_sizes[0], hidden_size, num_mlp_layers)])
        elif(n_keys == 2):
            self.in_mlps = nn.ModuleList([get_mlp(input_sizes[0], hidden_size, num_mlp_layers),
                                          get_mlp(input_sizes[1], hidden_size, num_mlp_layers)])
        elif(n_keys == 3):
            self.in_mlps = nn.ModuleList([get_mlp(input_sizes[0], hidden_size, num_mlp_layers),
                                          get_mlp(input_sizes[1], hidden_size, num_mlp_layers),
                                          get_mlp(input_sizes[2], hidden_size, num_mlp_layers)])
        else:
            raise ValueError("Expected 2 or 3 input keys. got: {}".format(input_keys))

        #Generate GRU: input_size = n_keys * hidden_size ; n_layers = 1 (that's the default config)
        self.gru = get_gru(n_keys, hidden_size, num_layers=num_gru_layers)

        #Generate output MLP: in_size: hidden_size + 2 ; n_layers = 3
        self.out_mlp = get_mlp(hidden_size, hidden_size, 3)

        self.proj_matrices = []
        self.output_keys = output_keys
        self.output_sizes = output_sizes
        for v,k in enumerate(output_keys):
            if k == 'f0_hz':
                self.proj_matrices.append(
                    nn.Sequential(
                        nn.Linear(hidden_size, output_sizes[v]),
                        FreqMap()
                    )
                )
            # elif k == "noise_bands":
            #     # initialize noise bands to 100x less than their original value
            #     self.proj_matrices.append(nn.Linear(hidden_size,output_sizes[v]))
            #     self.proj_matrices[-1].weight.data = self.proj_matrices[-1].weight.data * 0.01
            else:
                self.proj_matrices.append(nn.Linear(hidden_size,output_sizes[v]))

        self.proj_matrices = nn.ModuleList(self.proj_matrices)
        self.sample_rate = sample_rate

    def forward(self, x):
        # Run pitch and loudness and z (if available) inputs through the respectives input MLPs.
        # Then, concatenate the outputs in a flat vector.

        # Run through input_keys and load inputs accordingly
        hidden = torch.cat([self.in_mlps[v](x[k]) for v,k in enumerate(self.input_keys)],-1)

        # Run the flattened vector through the GRU.
        # The GRU predicts the embedding.
        # Then, concatenate the embedding with the disentangled parameters of pitch and loudness (nhid+2 size vector)
        # hidden = torch.cat([self.gru(hidden)[0], x['f0_scaled'], x['loudness_scaled']], -1)
        hidden = self.gru(hidden)[0] # HUGO: don't need the above line since we don'thave any disentangled params

        # Run the embedding through the output MLP to obtain a 512-sized output vector.
        hidden = self.out_mlp(hidden)

        # Run embedding through a projection_matrix to get outputs
        controls = {}
        for v,k in enumerate(self.output_keys):
            controls[k] = self.proj_matrices[v](hidden)

        return controls



from audiotools.ml.layers import BaseModel
class DDSPDAC(BaseModel):

    def __init__(self, encoder: DAC):
        super(DDSPDAC, self).__init__()

        self.encoder = encoder
        self.sample_rate = self.encoder.sample_rate
        self.block_size = self.encoder.hop_length

        self.hidden = self.encoder.latent_dim
        
        synth = HNSynth(self.sample_rate, self.block_size)
        decoder = RnnFCDecoder(
            sample_rate=self.sample_rate,
            input_keys=["latents"], 
            input_sizes=[self.hidden], 
            hidden_size=512, 
            output_keys=['amplitude','f0_hz', 'harmonic_distribution','noise_bands'],
            output_sizes=[1,1,100,65],
            num_gru_layers=3, 
            num_mlp_layers=3
        )

        self.decoder = DDSP_Decoder(decoder, synth)

    def preprocess(self, audio_data, sample_rate):
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate

        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.encoder.hop_length) * self.encoder.hop_length - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))

        return audio_data


    def forward(self, x, sample_rate):
        assert sample_rate == self.sample_rate
        latents = self.encoder.encode(x)["z"]
        # NOTE: the permute here is because the decoder (ddsp)
        # expects the latents in the shape (seq, batch, features)
        # instead of the regular shape (batch, features, seq)
        # which is used throughout this codebase. 
        latents = latents.permute(0, 2, 1) # (now seq, batch, features)
        out =  self.decoder({"latents":latents})
        return {"audio":out['synth_audio'].permute(0,2,1)}

if __name__ == "__main__":
    from audiotools import AudioSignal


    # load dac from pretrained
    dac = DAC.load("/home/hugo/.cache/descript/dac/weights_44khz_8kbps_0.0.1.pth")
    # del dac.decoder
    dac.eval()

    # load synth from pretrained
    model = DDSPDAC.load("runs/synth-debug-v5-justmel-vctk/latest/ddspdac/weights.pth",dac)
    # or alternatively, create a new model
    # model = DDSPDAC(dac)
    model.eval()

    # load audio
    sig = AudioSignal("data-fast/Clack/Clack_1/T2_BUCKEYE_FLY_BY_CS.wav")
    sig.resample(dac.sample_rate)
    sig.normalize(-16)
    sig.write('input.wav')
    import torch
    sig.samples = torch.cat([sig.samples, torch.zeros_like(sig.samples)])

    with torch.inference_mode():
        x = dac.preprocess(sig.samples, sig.sample_rate)
        y = model(x, dac.sample_rate)["audio"]

        recons = dac(x)["audio"]
        reconsig = AudioSignal(recons.detach(), dac.sample_rate)
        reconsig.write('recons.wav')

        sig = AudioSignal(y, dac.sample_rate)
        sig.write('out.wav')



    
