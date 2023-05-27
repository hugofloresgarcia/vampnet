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


def vamp(data):
    print(data[input_audio])
    sig = at.AudioSignal(data[input_audio])

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
    mask = pmask.dropout(mask, data[dropout])
    mask = pmask.codebook_unmask(mask, ncc)

    print(f"created mask with: linear random {data[rand_mask_intensity]}, inpaint {data[prefix_s]}:{data[suffix_s]}, periodic {data[periodic_p]}:{data[periodic_w]}, dropout {data[dropout]}")

    zv, mask_z = interface.coarse_vamp(
        z, 
        mask=mask,
        sampling_steps=data[num_steps],
        temperature=(data[init_temp], data[final_temp]),
        return_mask=True
    )

    if use_coarse2fine: 
        zv = interface.coarse_to_fine(zv)


    mask = interface.to_signal(mask_z).cpu()

    sig = interface.to_signal(zv).cpu()
    print("done")

    out_dir = OUT_DIR / str(uuid.uuid4())
    out_dir.mkdir()

    sig.write(out_dir / "output.wav")
    mask.write(out_dir / "mask.wav")
    return sig.path_to_file, mask.path_to_file
        
def save_vamp(data):
    out_dir = OUT_DIR / "saved" / str(uuid.uuid4())
    out_dir.mkdir(parents=True, exist_ok=True)

    sig_in = at.AudioSignal(input_audio)
    sig_out = at.AudioSignal(output_audio)

    sig_in.write(out_dir / "input.wav")
    sig_out.write(out_dir / "output.wav")
    
    data = {
        "init_temp": data[init_temp],
        "final_temp": data[final_temp],
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
        yaml.dump(data, f)

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
                value=9, 
            )
            periodic_w = gr.Slider(
                label="periodic prompt width (steps, 1 step ~= 10milliseconds)",
                minimum=1,
                maximum=20,
                step=1,
                value=1,
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

            with gr.Accordion("temperature settings", open=False):
                init_temp = gr.Slider(
                    label="initial temperature (should probably stay between 0.6 and 1)",
                    minimum=0.0,
                    maximum=1.5,
                    value=0.8
                )
                final_temp = gr.Slider(
                    label="final temperature (should probably stay between 0.7 and 2)",
                    minimum=0.0,
                    maximum=2.0,
                    value=1.0
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

            vamp_button = gr.Button("vamp!!!")

            output_audio = gr.Audio(
                label="output audio",
                interactive=False,
                type="filepath"
            )

        
        # with gr.Column():
        #     with gr.Accordion(label="beat unmask (how much time around the beat should be hinted?)"):
        #         use_beats = gr.Checkbox(
        #             label="use beat hints (helps the output stick to the beat structure of the input)",
        #             value=False
        #         )

        #         snap_to_beats = gr.Checkbox(
        #             label="trim to beat markers (uncheck if the output audio is too short.)",
        #             value=True
        #         )
                
        #         beat_unmask_dur = gr.Slider(
        #             label="duration", 
        #             minimum=0.0,
        #             maximum=3.0,
        #             value=0.07
        #         )


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
            
            thank_you = gr.Markdown("")
  
    # connect widgets
    vamp_button.click(
        fn=vamp,
        inputs={
            input_audio, 
            num_steps,
            init_temp, final_temp,
            prefix_s, suffix_s, 
            rand_mask_intensity, 
            periodic_p, periodic_w,
            n_conditioning_codebooks, 
            dropout,
            use_coarse2fine, 
            stretch_factor
        },
        outputs=[output_audio, audio_mask], 
        api_name="vamp"
    )

    save_button.click(
        fn=save_vamp,
        inputs={
            input_audio, 
            num_steps,
            init_temp, final_temp,
            prefix_s, suffix_s, 
            rand_mask_intensity, 
            periodic_p, periodic_w,
            n_conditioning_codebooks,
            dropout,
            use_coarse2fine, 
            stretch_factor, 
            notes_text
        },
        outputs=[thank_you, download_file]
    )

demo.launch(share=True, enable_queue=False, debug=True)
