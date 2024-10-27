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

import gradio as gr
from vampnet.interface import Interface, signal_concat
from vampnet import mask as pmask

device = "cuda" if torch.cuda.is_available() else "cpu"

interface = Interface.default()
init_model_choice = open("DEFAULT_MODEL").read().strip()
# load the init model
interface.load_finetuned(init_model_choice)
    
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


@spaces.GPU
def _vamp(
        seed, input_audio, model_choice, 
        pitch_shift_amt, periodic_p, 
        n_mask_codebooks, periodic_w, onset_mask_width, 
        dropout, sampletemp, typical_filtering, 
        typical_mass, typical_min_tokens, top_p, 
        sample_cutoff, stretch_factor, api=False
    ):

    t0 = time.time()
    interface.to("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device {interface.device}")
    _seed = seed if seed > 0 else None
    if _seed is None:
        _seed = int(torch.randint(0, 2**32, (1,)).item())
    at.util.seed(_seed)

    sr, input_audio = input_audio
    input_audio = input_audio / np.iinfo(input_audio.dtype).max
    
    sig = at.AudioSignal(input_audio, sr)

    # reload the model if necessary
    interface.load_finetuned(model_choice)

    if pitch_shift_amt != 0:
        sig = shift_pitch(sig, pitch_shift_amt)

    codes = interface.encode(sig)

    mask = interface.build_mask(
        codes, sig,
        rand_mask_intensity=1.0,
        prefix_s=0.0,
        suffix_s=0.0,
        periodic_prompt=int(periodic_p),
        periodic_prompt_width=periodic_w,
        onset_mask_width=onset_mask_width,
        _dropout=dropout,
        upper_codebook_mask=int(n_mask_codebooks), 
    )


    # save the mask as a txt file
    interface.set_chunk_size(10.0)
    codes, mask = interface.vamp(
        codes, mask,
        batch_size=1 if api else 1,
        feedback_steps=1,
        _sampling_steps=12 if sig.duration <6.0 else 24,
        time_stretch_factor=stretch_factor,
        return_mask=True,
        temperature=sampletemp,
        typical_filtering=typical_filtering, 
        typical_mass=typical_mass, 
        typical_min_tokens=typical_min_tokens, 
        top_p=None,
        seed=_seed,
        sample_cutoff=1.0,
    )
    print(f"vamp took {time.time() - t0} seconds")

    sig = interface.decode(codes)

    return to_output(sig)

def vamp(data):
    return _vamp(
        seed=data[seed], 
        input_audio=data[input_audio],
        model_choice=data[model_choice],
        pitch_shift_amt=data[pitch_shift_amt],
        periodic_p=data[periodic_p],
        n_mask_codebooks=data[n_mask_codebooks],
        periodic_w=data[periodic_w],
        onset_mask_width=data[onset_mask_width],
        dropout=data[dropout],
        sampletemp=data[sampletemp],
        typical_filtering=data[typical_filtering],
        typical_mass=data[typical_mass],
        typical_min_tokens=data[typical_min_tokens],
        top_p=data[top_p],
        sample_cutoff=data[sample_cutoff],
        stretch_factor=data[stretch_factor],
        api=False, 
    )

def api_vamp(data):
    return _vamp(
        seed=data[seed], 
        input_audio=data[input_audio],
        model_choice=data[model_choice],
        pitch_shift_amt=data[pitch_shift_amt],
        periodic_p=data[periodic_p],
        n_mask_codebooks=data[n_mask_codebooks],
        periodic_w=data[periodic_w],
        onset_mask_width=data[onset_mask_width],
        dropout=data[dropout],
        sampletemp=data[sampletemp],
        typical_filtering=data[typical_filtering],
        typical_mass=data[typical_mass],
        typical_min_tokens=data[typical_min_tokens],
        top_p=data[top_p],
        sample_cutoff=data[sample_cutoff],
        stretch_factor=data[stretch_factor],
        api=True, 
    )

OUT_DIR = Path("gradio-outputs")
OUT_DIR.mkdir(exist_ok=True)
def harp_vamp(input_audio_file, periodic_p, n_mask_codebooks, pitch_shift_amt):
    sig = at.AudioSignal(input_audio_file)
    sr, samples = sig.sample_rate, sig.samples[0][0].detach().cpu().numpy()
    # convert to int32
    samples = (samples * np.iinfo(np.int32).max).astype(np.int32)
    sr, samples =  _vamp(
        seed=0,
        input_audio=(sr, samples),
        model_choice=init_model_choice,
        pitch_shift_amt=pitch_shift_amt,
        periodic_p=periodic_p,
        n_mask_codebooks=n_mask_codebooks,
        periodic_w=1,
        onset_mask_width=0,
        dropout=0.0,
        sampletemp=1.0,
        typical_filtering=True,
        typical_mass=0.15,  
        typical_min_tokens=64,
        top_p=0.0,
        sample_cutoff=1.0,
        stretch_factor=1,
    )
    
    sig = at.AudioSignal(samples, sr)
    # write to file
    # clear the outdir
    for p in OUT_DIR.glob("*"):
        p.unlink()
    OUT_DIR.mkdir(exist_ok=True)
    outpath = OUT_DIR / f"{uuid.uuid4()}.wav"
    sig.write(outpath)
    from pyharp import OutputLabel
    output_labels = list()
    output_labels.append(OutputLabel(label='~', t=0.0, description='generated audio'))
    return outpath, output_labels
    

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

            audio_mask = gr.Audio(
                label="audio mask (listen to this to hear the mask hints)",
                interactive=False, 
                type="numpy",
            )

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
                    label="onset mask width (multiplies with the periodic mask, 1 step ~= 10milliseconds) ",
                    minimum=0,
                    maximum=100,
                    step=1,
                    value=0, visible=False
                )

                n_mask_codebooks = gr.Slider(
                    label="compression prompt ",
                    value=3,
                    minimum=1, 
                    maximum=14,
                    step=1,
                )
            
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

                periodic_w = gr.Slider(
                    label="periodic prompt width (steps, 1 step ~= 10milliseconds)",
                    minimum=1,
                    maximum=20,
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


            dropout = gr.Slider(
                label="mask dropout",
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=0.0
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
            for i in range(1):
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


    _inputs = {
            input_audio, 
            sampletemp,
            top_p,
            periodic_p, periodic_w,
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
        }
  
    # connect widgets
    vamp_button.click(
        fn=vamp,
        inputs=_inputs,
        outputs=[audio_outs[0]], 
    )

    api_vamp_button = gr.Button("api vamp", visible=True)
    api_vamp_button.click(
        fn=api_vamp,
        inputs=_inputs, 
        outputs=[audio_outs[0]], 
        api_name="vamp"
    )

    from pyharp import ModelCard, build_endpoint
    card = ModelCard(
        name="vampnet", 
        description="vampnet is a model for generating audio from audio",
        author="hugo flores garcía", 
        tags=["music generation"], 
        midi_in=False, 
        midi_out=False
    )
        
    # Build a HARP-compatible endpoint
    app = build_endpoint(model_card=card,
                         components=[
                            periodic_p, 
                            n_mask_codebooks,
                        ],
                         process_fn=harp_vamp)



try:
    demo.queue()
    demo.launch(share=True)
except KeyboardInterrupt:
    shutil.rmtree("gradio-outputs", ignore_errors=True)
    raise