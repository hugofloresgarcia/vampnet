from pathlib import Path
from typing import Tuple
import yaml
import tempfile
import uuid
from dataclasses import dataclass, asdict

import numpy as np
import audiotools as at
import argbind

import gradio as gr
from vampnet.interface import Interface
from vampnet import mask as pmask

Interface = argbind.bind(Interface)
AudioLoader = argbind.bind(at.data.datasets.AudioLoader)

conf = argbind.parse_args()

with argbind.scope(conf):
    interface = Interface()
    loader = AudioLoader()
    print(f"interface device is {interface.device}")

dataset = at.data.datasets.AudioDataset(
    loader,
    sample_rate=interface.codec.sample_rate,
    duration=interface.coarse.chunk_size_s,
    n_examples=5000,
    without_replacement=True,
)


checkpoints = {
    "spotdl": {
        "coarse": "./models/spotdl/coarse.pth",
        "c2f": "./models/spotdl/c2f.pth",
        "codec": "./models/spotdl/codec.pth",
        "full_ckpt": True
    }, 
    "berta": {
        "coarse": "./models/finetuned/berta-goldman-speech/coarse.pth",
        "c2f": "./models/finetuned/berta-goldman-speech/c2f.pth",
        "codec": "./model/spotdl/codec.pth",
        "full_ckpt": True
    }, 
    "xeno-canto-2": {
        "coarse": "./models/finetuned/xeno-canto-2/coarse.pth",
        "c2f": "./models/finetuned/xeno-canto-2/c2f.pth",
        "codec": "./models/spotdl/codec.pth",
        "full_ckpt": True
    }, 
    "panchos": {
        "coarse": "./models/finetuned/panchos/coarse.pth",
        "c2f": "./models/finetuned/panchos/c2f.pth",
        "codec": "./models/spotdl/codec.pth",
        "full_ckpt": False
    },
    "tv-choir": {
        "coarse": "./models/finetuned/tv-choir/coarse.pth",
        "c2f": "./models/finetuned/tv-choir/c2f.pth",
        "codec": "./models/spotdl/codec.pth",
        "full_ckpt": False
    }, 
    "titi": {
        "coarse": "./models/finetuned/titi/coarse.pth",
        "c2f": "./models/finetuned/titi/c2f.pth",
        "codec": "./models/spotdl/codec.pth",
        "full_ckpt": False
    }, 
    "titi-clean": {
        "coarse": "./models/finetuned/titi-clean/coarse.pth",
        "c2f": "./models/finetuned/titi-clean/c2f.pth",
        "codec": "./models/spotdl/codec.pth",
        "full_ckpt": False
    }, 
    "breaks-steps": {
        "coarse": "./models/finetuned/breaks-steps/coarse.pth",
        "c2f": None, #"./models/finetuned/breaks-steps/c2f.pth",
        "codec": "./models/spotdl/codec.pth",
        "full_ckpt": False
    },
}
interface.checkpoint_key = "spotdl"


OUT_DIR = Path("gradio-outputs")
OUT_DIR.mkdir(exist_ok=True, parents=True)


def load_audio(file):
    print(file)
    filepath = file.name
    sig = at.AudioSignal.salient_excerpt(
        filepath, 
        duration=interface.coarse.chunk_size_s
    )
    sig = interface.preprocess(sig)

    out_dir = OUT_DIR / "tmp" / str(uuid.uuid4())
    out_dir.mkdir(parents=True, exist_ok=True)
    sig.write(out_dir / "input.wav")
    return sig.path_to_file


def load_random_audio():
    index = np.random.randint(0, len(dataset))
    sig = dataset[index]["signal"]
    sig = interface.preprocess(sig)

    out_dir = OUT_DIR / "tmp" / str(uuid.uuid4())
    out_dir.mkdir(parents=True, exist_ok=True)
    sig.write(out_dir / "input.wav")
    return sig.path_to_file


def _vamp(data, return_mask=False):

    # if our checkpoint key is different, we need to load a new checkpoint
    if data[checkpoint_key] != interface.checkpoint_key:
        print(f"loading checkpoint {data[checkpoint_key]}")
        interface.lora_load(
            checkpoints[data[checkpoint_key]]["coarse"],
            checkpoints[data[checkpoint_key]]["c2f"],
            checkpoints[data[checkpoint_key]]["full_ckpt"],
        )
        interface.checkpoint_key = data[checkpoint_key]

    out_dir = OUT_DIR / str(uuid.uuid4())
    out_dir.mkdir()
    sig = at.AudioSignal(data[input_audio])

    # TODO: random pitch shift of segments in the signal to prompt! window size should be a parameter, pitch shift width should be a parameter

    z = interface.encode(sig)

    ncc = data[n_conditioning_codebooks]

    # build the mask
    mask = pmask.linear_random(z, data[rand_mask_intensity])
    mask = pmask.mask_and(
        mask, pmask.inpaint(
            z,
            interface.s2t(data[prefix_s]),
            interface.s2t(data[suffix_s])
        )
    )
    mask = pmask.mask_and(
        mask, pmask.periodic_mask(
            z,
            data[periodic_p],
            data[periodic_w],
            random_roll=True
        )
    )
    if data[onset_mask_width] > 0:
        mask = pmask.mask_or(
            mask, pmask.onset_mask(sig, z, interface, width=data[onset_mask_width])
        )
    if data[beat_mask_width] > 0:
        beat_mask = interface.make_beat_mask(
            sig,
            before_beat_s=(data[beat_mask_width]/1000)/2, 
            after_beat_s=(data[beat_mask_width]/1000)/2, 
            mask_upbeats=not data[beat_mask_downbeats],
        )
        mask = pmask.mask_and(mask, beat_mask)

    # these should be the last two mask ops
    mask = pmask.dropout(mask, data[dropout])
    mask = pmask.codebook_unmask(mask, ncc)


    print(f"created mask with: linear random {data[rand_mask_intensity]}, inpaint {data[prefix_s]}:{data[suffix_s]}, periodic {data[periodic_p]}:{data[periodic_w]}, dropout {data[dropout]}, codebook unmask {ncc}, onset mask {data[onset_mask_width]}, num steps {data[num_steps]}, init temp {data[temp]},  use coarse2fine {data[use_coarse2fine]}")
    # save the mask as a txt file
    np.savetxt(out_dir / "mask.txt", mask[:,0,:].long().cpu().numpy())

    zv, mask_z = interface.coarse_vamp(
        z, 
        mask=mask,
        sampling_steps=data[num_steps],
        temperature=data[temp]*10,
        return_mask=True, 
        typical_filtering=data[typical_filtering], 
        typical_mass=data[typical_mass], 
        typical_min_tokens=data[typical_min_tokens], 
        gen_fn=interface.coarse.generate,
    )

    if use_coarse2fine: 
        zv = interface.coarse_to_fine(zv, temperature=data[temp])

    sig = interface.to_signal(zv).cpu()
    print("done")

    

    sig.write(out_dir / "output.wav")

    if return_mask:
        mask = interface.to_signal(mask_z).cpu()
        mask.write(out_dir / "mask.wav")
        return sig.path_to_file, mask.path_to_file
    else:
        return sig.path_to_file

def vamp(data):
    return _vamp(data, return_mask=True)

def api_vamp(data):
    return _vamp(data, return_mask=False)
        
def save_vamp(data):
    out_dir = OUT_DIR / "saved" / str(uuid.uuid4())
    out_dir.mkdir(parents=True, exist_ok=True)

    sig_in = at.AudioSignal(data[input_audio])
    sig_out = at.AudioSignal(data[output_audio])

    sig_in.write(out_dir / "input.wav")
    sig_out.write(out_dir / "output.wav")
    
    _data = {
        "temp": data[temp],
        "prefix_s": data[prefix_s],
        "suffix_s": data[suffix_s],
        "rand_mask_intensity": data[rand_mask_intensity],
        "num_steps": data[num_steps],
        "notes": data[notes_text],
        "periodic_period": data[periodic_p],
        "periodic_width": data[periodic_w],
        "n_conditioning_codebooks": data[n_conditioning_codebooks], 
        "use_coarse2fine": data[use_coarse2fine],
        "stretch_factor": data[stretch_factor],
    }

    # save with yaml
    with open(out_dir / "data.yaml", "w") as f:
        yaml.dump(_data, f)

    import zipfile
    zip_path = out_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        for file in out_dir.iterdir():
            zf.write(file, file.name)

    return f"saved! your save code is {out_dir.stem}", zip_path



with gr.Blocks() as demo:

    with gr.Row():
        with gr.Column():
            use_coarse2fine = gr.Checkbox(
                label="use coarse2fine",
                value=True
            )

            manual_audio_upload = gr.File(
                label=f"upload some audio (will be randomly trimmed to max of {interface.coarse.chunk_size_s:.2f}s)",
                file_types=["audio"]
            )
            load_random_audio_button = gr.Button("or load random audio")

            input_audio = gr.Audio(
                label="input audio",
                interactive=False, 
                type="filepath",
            )

            audio_mask = gr.Audio(
                label="audio mask (listen to this to hear the mask hints)",
                interactive=False, 
                type="filepath",
            )

            # connect widgets
            load_random_audio_button.click(
                fn=load_random_audio,
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

            input_pitch_shift = gr.Slider(
                label="input pitch shift (semitones)",
                minimum=-36,
                maximum=36,
                step=1,
                value=0,
            )

            rand_mask_intensity = gr.Slider(
                label="random mask intensity. (If this is less than 1, scatters prompts throughout the audio, should be between 0.9 and 1.0)",
                minimum=0.0,
                maximum=1.0,
                value=1.0
            )

            periodic_p = gr.Slider(
                label="periodic prompt  (0.0 means no hint, 2 - lots of hints, 8 - a couple of hints, 16 - occasional hint, 32 - very occasional hint, etc)",
                minimum=0,
                maximum=128, 
                step=1,
                value=3, 
            )
            periodic_w = gr.Slider(
                label="periodic prompt width (steps, 1 step ~= 10milliseconds)",
                minimum=1,
                maximum=20,
                step=1,
                value=1,
            )

            onset_mask_width = gr.Slider(
                label="onset mask width (steps, 1 step ~= 10milliseconds)",
                minimum=0,
                maximum=20,
                step=1,
                value=5,
            )

            beat_mask_width = gr.Slider(
                label="beat mask width (in milliseconds)",
                minimum=0,
                maximum=200,
                value=0,
            )
            beat_mask_downbeats = gr.Checkbox(
                label="beat mask downbeats only?", 
                value=False
            )


            with gr.Accordion("extras ", open=False):
                n_conditioning_codebooks = gr.Number(
                    label="number of conditioning codebooks. probably 0", 
                    value=0,
                    precision=0,
                )

                stretch_factor = gr.Slider(
                    label="time stretch factor",
                    minimum=0,
                    maximum=64, 
                    step=1,
                    value=1, 
                )


            with gr.Accordion("prefix/suffix hints", open=False):
                prefix_s = gr.Slider(
                    label="prefix hint length (seconds)",
                    minimum=0.0,
                    maximum=10.0,
                    value=0.0
                )
                suffix_s = gr.Slider(
                    label="suffix hint length (seconds)",
                    minimum=0.0,
                    maximum=10.0,
                    value=0.0
                )

            temp = gr.Slider(
                label="temperature",
                minimum=0.0,
                maximum=3.0,
                value=0.8
            )

            with gr.Accordion("sampling settings", open=False):
                typical_filtering = gr.Checkbox(
                    label="typical filtering ",
                    value=False
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

            num_steps = gr.Slider(
                label="number of steps (should normally be between 12 and 36)",
                minimum=1,
                maximum=128,
                step=1,
                value=36
            )

            dropout = gr.Slider(
                label="mask dropout",
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=0.0
            )


        # mask settings
        with gr.Column():
            checkpoint_key = gr.Radio(
                label="checkpoint",
                choices=list(checkpoints.keys()),
                value="spotdl" 
            )
            vamp_button = gr.Button("vamp!!!")
            output_audio = gr.Audio(
                label="output audio",
                interactive=False,
                type="filepath"
            )

            notes_text = gr.Textbox(
                label="type any notes about the generated audio here", 
                value="",
                interactive=True
            )
            save_button = gr.Button("save vamp")
            download_file = gr.File(
                label="vamp to download will appear here",
                interactive=False
            )
            use_as_input_button = gr.Button("use output as input")
            
            thank_you = gr.Markdown("")


    _inputs = {
            input_audio, 
            num_steps,
            temp,
            prefix_s, suffix_s, 
            rand_mask_intensity, 
            periodic_p, periodic_w,
            n_conditioning_codebooks, 
            dropout,
            use_coarse2fine, 
            stretch_factor, 
            onset_mask_width, 
            input_pitch_shift, 
            typical_filtering,
            typical_mass,
            typical_min_tokens,
            checkpoint_key,
            beat_mask_width,
            beat_mask_downbeats
        }
  
    # connect widgets
    vamp_button.click(
        fn=vamp,
        inputs=_inputs,
        outputs=[output_audio, audio_mask], 
    )

    api_vamp_button = gr.Button("api vamp")
    api_vamp_button.click(
        fn=api_vamp,
        inputs=_inputs, 
        outputs=[output_audio], 
        api_name="vamp"
    )

    use_as_input_button.click(
        fn=lambda x: x,
        inputs=[output_audio],
        outputs=[input_audio]
    )

    save_button.click(
        fn=save_vamp,
        inputs=_inputs | {notes_text, output_audio},
        outputs=[thank_you, download_file]
    )

demo.launch(share=True, enable_queue=False, debug=True, server_name="0.0.0.0")
