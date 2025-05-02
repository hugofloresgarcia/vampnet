import spaces
from pathlib import Path
import yaml
import time
import uuid

import numpy as np
import audiotools as at
import argbind
import shutil
import torch
from datetime import datetime
from pyharp import load_audio, save_audio, OutputLabel, LabelList, build_endpoint, ModelCard

import gradio as gr
from vampnet.interface import Interface, signal_concat
from vampnet import mask as pmask

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"using device {device}\n"*10)

interface = Interface.default()
init_model_choice = open("DEFAULT_MODEL").read().strip()

# load the init model
interface.load_finetuned(init_model_choice)
interface.to(device)
    
def to_output(sig):
    return sig.sample_rate, sig.cpu().detach().numpy()[0][0]

MAX_DURATION_S = 10
def load_audio(file):
    print(file)
    if isinstance(file, str):
        filepath = file
    elif isinstance(file, tuple):
        # not a file
        sr, samples = file
        samples = samples / np.iinfo(samples.dtype).max
        return sr, samples
    else:
        filepath = file.name
    sig = at.AudioSignal.salient_excerpt(
        filepath, duration=MAX_DURATION_S
    )
    sig = at.AudioSignal(filepath)
    return to_output(sig)


def load_example_audio():
    return load_audio("./assets/example.wav")

from torch_pitch_shift import pitch_shift, get_fast_shifts
def shift_pitch(signal, interval: int):
    signal.samples = pitch_shift(
        signal.samples, 
        shift=interval, 
        sample_rate=signal.sample_rate
    )
    return signal


def onsets(sig: at.AudioSignal, hop_length: int):
    assert sig.batch_size == 1, "batch size must be 1"
    assert sig.num_channels == 1, "mono signals only"
    import librosa
    onset_frame_idxs = librosa.onset.onset_detect(
        y=sig.samples[0][0].detach().cpu().numpy(), sr=sig.sample_rate, 
        hop_length=hop_length,
        backtrack=True,
    )
    return onset_frame_idxs


def new_vampnet_mask(self, 
    codes, 
    onset_idxs, 
    width: int = 5, 
    periodic_prompt=2, 
    upper_codebook_mask=1,
    drop_amt: float = 0.1
):
    from vampnet.newmask import mask_and, mask_or, onset_mask, periodic_mask, drop_ones, codebook_mask
    mask =  mask_and(
        periodic_mask(codes, periodic_prompt, 1, random_roll=False),
        mask_or( # this re-masks the onsets, according to a periodic schedule
            onset_mask(onset_idxs, codes, width=width),
            periodic_mask(codes, periodic_prompt, 1, random_roll=False),
        )
    ).int()
    # make sure the onset idxs themselves are unmasked
    # mask = 1 - mask
    mask[:, :, onset_idxs] = 0
    mask = mask.cpu() # debug
    mask = 1-drop_ones(1-mask, drop_amt)
    mask = codebook_mask(mask, upper_codebook_mask)

    
    # save mask as txt (ints)
    np.savetxt("scratch/rms_mask.txt", mask[0].cpu().numpy(), fmt='%d')
    mask = mask.to(self.device)
    return mask[:, :, :]

def mask_preview(periodic_p, n_mask_codebooks, onset_mask_width, dropout):
    # make a mask preview
    codes = torch.zeros((1, 14, 80)).to(device)
    mask = interface.build_mask(
        codes,
        periodic_prompt=periodic_p,
        # onset_mask_width=onset_mask_width,
        _dropout=dropout,
        upper_codebook_mask=n_mask_codebooks,
    )
    # mask = mask.cpu().numpy()
    import matplotlib.pyplot as plt
    plt.clf()
    interface.visualize_codes(mask)
    plt.title("mask preview")
    plt.savefig("scratch/mask-prev.png")
    return "scratch/mask-prev.png"


@spaces.GPU
def _vamp_internal(
        seed, input_audio, model_choice, 
        pitch_shift_amt, periodic_p, 
        n_mask_codebooks, onset_mask_width, 
        dropout, sampletemp, typical_filtering, 
        typical_mass, typical_min_tokens, top_p, 
        sample_cutoff, stretch_factor, sampling_steps, beat_mask_ms, num_feedback_steps, api=False
    ):

    print("args!")
    print(f"seed: {seed}")
    print(f"input_audio: {input_audio}")
    print(f"model_choice: {model_choice}")
    print(f"pitch_shift_amt: {pitch_shift_amt}")
    print(f"periodic_p: {periodic_p}")
    print(f"n_mask_codebooks: {n_mask_codebooks}")
    print(f"onset_mask_width: {onset_mask_width}")
    print(f"dropout: {dropout}")
    print(f"sampletemp: {sampletemp}")
    print(f"typical_filtering: {typical_filtering}")
    print(f"typical_mass: {typical_mass}")
    print(f"typical_min_tokens: {typical_min_tokens}")
    print(f"top_p: {top_p}")
    print(f"sample_cutoff: {sample_cutoff}")
    print(f"stretch_factor: {stretch_factor}")
    print(f"sampling_steps: {sampling_steps}")
    print(f"api: {api}")
    print(f"beat_mask_ms: {beat_mask_ms}")
    print(f"using device {interface.device}")
    print(f"num feedback steps: {num_feedback_steps}")


    t0 = time.time()
    interface.to(device)
    print(f"using device {interface.device}")
    _seed = seed if seed > 0 else None
    if _seed is None:
        _seed = int(torch.randint(0, 2**32, (1,)).item())
    at.util.seed(_seed)

    if input_audio is None:
        raise gr.Error("no input audio received!")
    sr, input_audio = input_audio
    input_audio = input_audio / np.iinfo(input_audio.dtype).max
    
    sig = at.AudioSignal(input_audio, sr).to_mono()

    loudness = sig.loudness()
    sig = interface._preprocess(sig)

    # reload the model if necessary
    interface.load_finetuned(model_choice)

    if pitch_shift_amt != 0:
        sig = shift_pitch(sig, pitch_shift_amt)

    codes = interface.encode(sig)

    # mask = new_vampnet_mask(
    #     interface, 
    #     codes, 
    #     onset_idxs=onsets(sig, hop_length=interface.codec.hop_length),
    #     width=onset_mask_width,
    #     periodic_prompt=periodic_p,
    #     upper_codebook_mask=n_mask_codebooks,
    #     drop_amt=dropout
    # ).long()

    
    mask = interface.build_mask(
        codes,
        sig=sig, 
        periodic_prompt=periodic_p,
        onset_mask_width=onset_mask_width,
        _dropout=dropout,
        upper_codebook_mask=n_mask_codebooks,
    )
    if beat_mask_ms > 0:
        # bm = pmask.mask_or(
        #     pmask.periodic_mask(
        #         codes, periodic_p, random_roll=False
        #     ),
        # )
        mask = pmask.mask_and(
            mask, interface.make_beat_mask(
                sig, after_beat_s=beat_mask_ms/1000.,
            )
        )
        mask = pmask.codebook_mask(mask, n_mask_codebooks)
    np.savetxt("scratch/rms_mask.txt", mask[0].cpu().numpy(), fmt='%d')

    interface.set_chunk_size(10.0)

    # lord help me
    if top_p is not None:
        if top_p > 0:
            pass
        else:
            top_p = None

    codes, mask_z = interface.vamp(
        codes, mask,
        batch_size=2,
        feedback_steps=num_feedback_steps,
        _sampling_steps=sampling_steps,
        time_stretch_factor=stretch_factor,
        return_mask=True,
        temperature=sampletemp,
        typical_filtering=typical_filtering, 
        typical_mass=typical_mass, 
        typical_min_tokens=typical_min_tokens, 
        top_p=top_p,
        seed=_seed,
        sample_cutoff=sample_cutoff,
    )
    print(f"vamp took {time.time() - t0} seconds")

    sig = interface.decode(codes)
    sig = sig.normalize(loudness)

    import matplotlib.pyplot as plt
    plt.clf()
    # plt.imshow(mask_z[0].cpu().numpy(), aspect='auto
    interface.visualize_codes(mask)
    plt.title("actual mask")
    plt.savefig("scratch/mask.png")
    plt.clf()

    if not api:
        return to_output(sig[0]), to_output(sig[1]), "scratch/mask.png"
    else:
        return to_output(sig[0]), to_output(sig[1])


def vamp(input_audio, 
        sampletemp,
        top_p,
        periodic_p, 
        dropout,
        stretch_factor, 
        onset_mask_width, 
        typical_filtering,
        typical_mass,
        typical_min_tokens,
        seed, 
        model_choice,
        n_mask_codebooks,
        pitch_shift_amt, 
        sample_cutoff, 
        sampling_steps, 
        beat_mask_ms,
        num_feedback_steps):
    return _vamp_internal(
        seed=seed,
        input_audio=input_audio,
        model_choice=model_choice,
        pitch_shift_amt=pitch_shift_amt,
        periodic_p=periodic_p,
        n_mask_codebooks=n_mask_codebooks,
        onset_mask_width=onset_mask_width,
        dropout=dropout,
        sampletemp=sampletemp,
        typical_filtering=typical_filtering,
        typical_mass=typical_mass,
        typical_min_tokens=typical_min_tokens,
        top_p=top_p,
        sample_cutoff=sample_cutoff,
        stretch_factor=stretch_factor,
        sampling_steps=sampling_steps,
        beat_mask_ms=beat_mask_ms,
        num_feedback_steps=num_feedback_steps,
        api=False,
    )


def api_vamp(input_audio, 
                sampletemp, top_p, 
                periodic_p,
                dropout, 
                stretch_factor,
                onset_mask_width,
                typical_filtering,
                typical_mass,
                typical_min_tokens,
                seed,
                model_choice,
                n_mask_codebooks,
                pitch_shift_amt,
                sample_cutoff, 
                sampling_steps, 
                beat_mask_ms, num_feedback_steps):
    return _vamp_internal(
        seed=seed, 
        input_audio=input_audio,
        model_choice=model_choice,
        pitch_shift_amt=pitch_shift_amt,
        periodic_p=periodic_p,
        n_mask_codebooks=n_mask_codebooks,
        onset_mask_width=onset_mask_width,
        dropout=dropout,
        sampletemp=sampletemp,
        typical_filtering=typical_filtering,
        typical_mass=typical_mass,
        typical_min_tokens=typical_min_tokens,
        top_p=top_p,
        sample_cutoff=sample_cutoff,
        stretch_factor=stretch_factor,
        sampling_steps=sampling_steps,
        beat_mask_ms=beat_mask_ms,
        num_feedback_steps=num_feedback_steps,
        api=True, 
    )

def harp_vamp(input_audio, sampletemp, periodic_p, dropout, n_mask_codebooks, model_choice, beat_mask_ms):
    sig = at.AudioSignal(input_audio).to_mono()

    input_audio = sig.cpu().detach().numpy()[0][0]
    input_audio = input_audio * np.iinfo(np.int16).max
    input_audio = input_audio.astype(np.int16)
    input_audio = input_audio.reshape(1, -1)
    input_audio = (sig.sample_rate, input_audio)

    out =  _vamp_internal(
        seed=0, 
        input_audio=input_audio,
        model_choice=model_choice,
        pitch_shift_amt=0,
        periodic_p=int(periodic_p),
        n_mask_codebooks=int(n_mask_codebooks),
        onset_mask_width=0,
        dropout=dropout,
        sampletemp=sampletemp,
        typical_filtering=False,
        typical_mass=0.15,
        typical_min_tokens=1,
        top_p=None,
        sample_cutoff=1.0,
        stretch_factor=1.0,
        sampling_steps=36,
        beat_mask_ms=int(beat_mask_ms),
        num_feedback_steps=1
    )
    sr, output_audio = out
    # save the output audio
    sig = at.AudioSignal(output_audio, sr).to_mono()

    ll = LabelList()
    ll.append(OutputLabel(label='short label', t=0.0, description='longer description'))
    return save_audio(sig), ll


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            manual_audio_upload = gr.File(
                label=f"upload some audio (will be randomly trimmed to max of 100s)",
                file_types=["audio"]
            )
            load_example_audio_button = gr.Button("or load example audio")

            input_audio = gr.Audio(
                label="input audio",
                interactive=False, 
                type="numpy",
            )

            # audio_mask = gr.Audio(
            #     label="audio mask (listen to this to hear the mask hints)",
            #     interactive=False, 
            #     type="numpy",
            # )

            # connect widgets
            load_example_audio_button.click(
                fn=load_example_audio,
                inputs=[],
                outputs=[ input_audio]
            )

            manual_audio_upload.change(
                fn=load_audio,
                inputs=[manual_audio_upload],
                outputs=[ input_audio]
            )
                

        # mask settings
        with gr.Column():
            with gr.Accordion("manual controls", open=True):
                periodic_p = gr.Slider(
                    label="periodic prompt",
                    minimum=0,
                    maximum=13, 
                    step=1,
                    value=7, 
                )

                onset_mask_width = gr.Slider(
                    label="onset mask width (multiplies with the periodic mask, 1 step ~= 10milliseconds) does not affect mask preview",
                    minimum=0,
                    maximum=100,
                    step=1,
                    value=0, visible=True
                )

                beat_mask_ms = gr.Slider(
                    label="beat mask width (milliseconds) does not affect mask preview",
                    minimum=1,
                    maximum=200, 
                    step=1,
                    value=0, 
                    visible=True
                )

                n_mask_codebooks = gr.Slider(
                    label="compression prompt ",
                    value=3,
                    minimum=1, 
                    maximum=14,
                    step=1,
                )

                dropout = gr.Slider(
                    label="mask dropout",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=0.0
                )

                num_feedback_steps = gr.Slider(
                    label="feedback steps (token telephone) -- turn it up for better timbre/rhythm transfer quality, but it's slower!",
                    minimum=1,
                    maximum=8,
                    step=1,
                    value=1
                )

                preset_dropdown = gr.Dropdown(
                    label="preset",
                    choices=["timbre transfer", "small variation", "small variation (follow beat)", "medium variation", "medium variation (follow beat)", "large variation", "large variation (follow beat)", "unconditional"],
                    value="medium variation"
                )
                def change_preset(preset_dropdown):
                    if preset_dropdown == "timbre transfer":
                        periodic_p = 2
                        n_mask_codebooks = 1
                        onset_mask_width = 0
                        dropout = 0.0
                        beat_mask_ms = 0
                    elif preset_dropdown == "small variation":
                        periodic_p = 5
                        n_mask_codebooks = 4
                        onset_mask_width = 0
                        dropout = 0.0
                        beat_mask_ms = 0
                    elif preset_dropdown == "small variation (follow beat)":
                        periodic_p = 7
                        n_mask_codebooks = 4
                        onset_mask_width = 0
                        dropout = 0.0
                        beat_mask_ms = 50
                    elif preset_dropdown == "medium variation":
                        periodic_p = 7
                        n_mask_codebooks = 4
                        onset_mask_width = 0
                        dropout = 0.0
                        beat_mask_ms = 0
                    elif preset_dropdown == "medium variation (follow beat)":
                        periodic_p = 13
                        n_mask_codebooks = 4
                        onset_mask_width = 0
                        dropout = 0.0
                        beat_mask_ms = 50
                    elif preset_dropdown == "large variation":
                        periodic_p = 13
                        n_mask_codebooks = 4
                        onset_mask_width = 0
                        dropout = 0.2
                        beat_mask_ms = 0 
                    elif preset_dropdown == "large variation (follow beat)":
                        periodic_p = 0
                        n_mask_codebooks = 4
                        onset_mask_width = 0 
                        dropout = 0.0
                        beat_mask_ms=80 
                    elif preset_dropdown == "unconditional":
                        periodic_p=0
                        n_mask_codebooks=1
                        onset_mask_width=0 
                        dropout=0.0
                    return periodic_p, n_mask_codebooks, onset_mask_width, dropout, beat_mask_ms
                preset_dropdown.change(
                    fn=change_preset,
                    inputs=[preset_dropdown],
                    outputs=[periodic_p, n_mask_codebooks, onset_mask_width, dropout, beat_mask_ms]
                )
                # preset_dropdown.change(


            maskimg = gr.Image(
                label="mask image",
                interactive=False,
                type="filepath"
            )

            with gr.Accordion("extras ", open=False):
                pitch_shift_amt = gr.Slider(
                    label="pitch shift amount (semitones)",
                    minimum=-12,
                    maximum=12,
                    step=1,
                    value=0,
                )

                stretch_factor = gr.Slider(
                    label="time stretch factor",
                    minimum=0,
                    maximum=8, 
                    step=1,
                    value=1, 
                )




            with gr.Accordion("sampling settings", open=False):
                sampletemp = gr.Slider(
                    label="sample temperature",
                    minimum=0.1,
                    maximum=10.0,
                    value=1.0, 
                    step=0.001
                )
            
                top_p = gr.Slider(
                    label="top p (0.0 = off)",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.0
                )
                typical_filtering = gr.Checkbox(
                    label="typical filtering ",
                    value=True
                )
                typical_mass = gr.Slider( 
                    label="typical mass (should probably stay between 0.1 and 0.5)",
                    minimum=0.01,
                    maximum=0.99,
                    value=0.15
                )
                typical_min_tokens = gr.Slider(
                    label="typical min tokens (should probably stay between 1 and 256)",
                    minimum=1,
                    maximum=256,
                    step=1,
                    value=64
                )
                sample_cutoff = gr.Slider(
                    label="sample cutoff",
                    minimum=0.0,
                    maximum=0.9,
                    value=1.0, 
                    step=0.01
                )
                sampling_steps = gr.Slider(
                    label="sampling steps",
                    minimum=1,
                    maximum=128,
                    step=1,
                    value=36
                )



            seed = gr.Number(
                label="seed (0 for random)",
                value=0,
                precision=0,
            )


        # mask settings
        with gr.Column():

            model_choice = gr.Dropdown(
                label="model choice", 
                choices=list(interface.available_models()),
                value=init_model_choice, 
                visible=True
            )


            vamp_button = gr.Button("generate (vamp)!!!")


            audio_outs = []
            use_as_input_btns = []
            for i in range(2):
                with gr.Column():
                    audio_outs.append(gr.Audio(
                        label=f"output audio {i+1}",
                        interactive=False,
                        type="numpy"
                    ))
                    use_as_input_btns.append(
                        gr.Button(f"use as input (feedback)")
                    )

            thank_you = gr.Markdown("")

            # download all the outputs
            # download = gr.File(type="filepath", label="download outputs")


    # mask preview change
    for widget in (
        periodic_p, n_mask_codebooks, 
        onset_mask_width, dropout
    ):
        widget.change(
            fn=mask_preview,
            inputs=[periodic_p, n_mask_codebooks, 
                    onset_mask_width, dropout],
            outputs=[maskimg]
        )


    _inputs = [
            input_audio, 
            sampletemp,
            top_p,
            periodic_p,
            dropout,
            stretch_factor, 
            onset_mask_width, 
            typical_filtering,
            typical_mass,
            typical_min_tokens,
            seed, 
            model_choice,
            n_mask_codebooks,
            pitch_shift_amt, 
            sample_cutoff, 
            sampling_steps, 
            beat_mask_ms,
            num_feedback_steps
    ]
  
    # connect widgets
    vamp_button.click(
        fn=vamp,
        inputs=_inputs,
        outputs=[audio_outs[0], audio_outs[1], maskimg], 
    )

    api_vamp_button = gr.Button("api vamp", visible=True)
    api_vamp_button.click(
        fn=api_vamp,
        inputs=[input_audio, 
                sampletemp, top_p, 
                periodic_p, 
                dropout, 
                stretch_factor,
                onset_mask_width,
                typical_filtering,
                typical_mass,
                typical_min_tokens,
                seed,
                model_choice,
                n_mask_codebooks,
                pitch_shift_amt,
                sample_cutoff, 
                sampling_steps, 
                beat_mask_ms,
                num_feedback_steps
        ], 
        outputs=[audio_outs[0], audio_outs[1]],
        api_name="vamp"
    )


    app = build_endpoint(
        model_card=ModelCard(
            name="vampnet",
            description="generating audio by filling in the blanks.",
            author="hugo flores garcía et al. (descript/northwestern)",
            tags=["sound", "generation",],
            midi_in=False,
            midi_out=False,
        ), 
        components=[
            sampletemp, periodic_p, dropout, n_mask_codebooks, model_choice, beat_mask_ms
        ],
        process_fn=harp_vamp,
    )

try:
    demo.queue()
    demo.launch(share=True)
except KeyboardInterrupt:
    shutil.rmtree("gradio-outputs", ignore_errors=True)
    raise