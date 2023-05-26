from pathlib import Path
from typing import Tuple
import yaml
import tempfile
import uuid

import numpy as np
import audiotools as at
import argbind

import gradio as gr
from vampnet.interface import Interface

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


def ez_vamp(
    input_audio, init_temp, final_temp,  
    mask_periodic_amt, mask_periodic_width, num_steps,  
    stretch_factor, 
):
    print(input_audio)
    sig = at.AudioSignal(input_audio)

    print(f"running standard vampnet with {num_vamps} vamps")
    zv = interface.coarse_vamp(
        sig, 
        sampling_steps=num_steps,
        temperature=(init_temp, final_temp),
        prefix_dur_s=0.0,
        suffix_dur_s=0.0,
        num_vamps=1,
        downsample_factor=mask_periodic_amt, 
        stretch_factor=stretch_factor,
        periodic_width=mask_periodic_width,
        periodic_dropout=0.0,
        periodic_width_dropout=0.0,
        n_conditioning_codebooks=None,
        intensity=1.0,
        ext_mask=None, 
    )

    zv = interface.coarse_to_fine(zv)

    sig = interface.to_signal(zv).cpu()
    print("done")

    out_dir = OUT_DIR / str(uuid.uuid4())
    out_dir.mkdir()
    sig.write(out_dir / "output.wav")
    # mask.write(out_dir / "mask.wav")
    # return sig.path_to_file, mask.path_to_file
    return sig.path_to_file


def vamp(
    input_audio, init_temp, final_temp, 
    prefix_s, suffix_s, rand_mask_intensity,
    mask_periodic_amt, beat_unmask_dur,
    mask_dwn_chk, dwn_factor,
    mask_up_chk, up_factor, 
    num_vamps, mode, use_beats, num_steps, snap_to_beats, 
    beat_unmask_drop,  mask_periodic_width, 
    mask_periodic_dropout, mask_periodic_width_dropout, 
    n_conditioning_codebooks, use_coarse2fine, stretch_factor, 
):
    # try:
        print(input_audio)

        sig = at.AudioSignal(input_audio)

        if snap_to_beats:
            old_sig = sig.clone()
            sig = interface.snap_to_beats(sig)
            if sig.duration < (sig.duration / 4): # we cut off too much
                sig = old_sig
                print(f"new sig duration is {sig.duration} which is too short, reverting to old sig")
            print(f"new sig duration is {sig.duration}")

        if beat_unmask_dur > 0.0 and use_beats:
            beat_mask = interface.make_beat_mask(
                sig,
                before_beat_s=0.0,
                after_beat_s=beat_unmask_dur,
                mask_downbeats=mask_dwn_chk,
                mask_upbeats=mask_up_chk,
                downbeat_downsample_factor=dwn_factor if dwn_factor > 0 else None, 
                beat_downsample_factor=up_factor if up_factor > 0 else None,
                dropout=beat_unmask_drop, 
                invert=True
            )
            print(beat_mask)
        else:
            beat_mask = None

        if mode == "standard": 
            print(f"running standard vampnet with {num_vamps} vamps")
            zv, mask_z = interface.coarse_vamp(
                sig, 
                sampling_steps=num_steps,
                temperature=(init_temp, final_temp),
                prefix_dur_s=prefix_s,
                suffix_dur_s=suffix_s,
                num_vamps=num_vamps,
                downsample_factor=mask_periodic_amt,
                stretch_factor=stretch_factor,
                periodic_width=mask_periodic_width,
                periodic_dropout=mask_periodic_dropout,
                periodic_width_dropout=mask_periodic_width_dropout,
                n_conditioning_codebooks=n_conditioning_codebooks if n_conditioning_codebooks > 0 else None,
                intensity=rand_mask_intensity,
                ext_mask=beat_mask, 
                verbose=True,
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
        # return sig.path_to_file, mask_z
    # except Exception as e:
    #     raise gr.Error(f"failed with error: {e}")
        
def save_vamp(
    input_audio, init_temp, final_temp, 
    prefix_s, suffix_s, rand_mask_intensity,
    mask_periodic_amt, beat_unmask_dur,
    mask_dwn_chk, dwn_factor,
    mask_up_chk, up_factor, 
    num_vamps, mode, output_audio, notes, use_beats, num_steps, snap_to_beats,
    beat_unmask_drop, mask_periodic_width, mask_periodic_dropout, mask_periodic_width_dropout, 
    n_conditioning_codebooks, use_coarse2fine, stretch_factor
):
    out_dir = OUT_DIR / "saved" / str(uuid.uuid4())
    out_dir.mkdir(parents=True, exist_ok=True)

    sig_in = at.AudioSignal(input_audio)
    sig_out = at.AudioSignal(output_audio)

    sig_in.write(out_dir / "input.wav")
    sig_out.write(out_dir / "output.wav")
    
    data = {
        "init_temp": init_temp,
        "final_temp": final_temp,
        "prefix_s": prefix_s,
        "suffix_s": suffix_s,
        "rand_mask_intensity": rand_mask_intensity,
        "mask_periodic_amt": mask_periodic_amt,
        "use_beats": use_beats,
        "beat_unmask_dur": beat_unmask_dur,
        "mask_dwn_chk": mask_dwn_chk,
        "dwn_factor": dwn_factor,
        "mask_up_chk": mask_up_chk,
        "up_factor": up_factor,
        "num_vamps": num_vamps,
        "num_steps": num_steps,
        "snap_to_beats": snap_to_beats,
        "mode": mode,
        "notes": notes,
        "beat_unmask_drop": beat_unmask_drop,
        "mask_periodic_width": mask_periodic_width, 
        "mask_periodic_dropout": mask_periodic_dropout,
        "mask_periodic_width_dropout": mask_periodic_width_dropout, 
        "n_conditioning_codebooks": n_conditioning_codebooks, 
        "use_coarse2fine": use_coarse2fine,
        "stretch_factor": stretch_factor,
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
        # input audio
        with gr.Column():
            gr.Markdown("""
            # Vampnet
            **Instructions**:
            1. Upload some audio (or click the load random audio button)
            2. Adjust the mask hints. The more hints, the more the generated music will follow the input music
            3. Adjust the vampnet parameters. The more vamps, the longer the generated music will be
            4. Click the "vamp" button
            5. Listen to the generated audio
            6. If you noticed something you liked, write some notes, click the "save vamp" button, and copy the save code

            """)
            gr.Markdown("## Input Audio")

        with gr.Column():
            gr.Markdown("""
            ### Tips
            - use the beat hint button so the output audio has the same beat structure as the input audio
            - if you want more beat structure:
                - enable beat hints
            - if you want a more "random" generation:
                - increase the periodic unmasking to 12 or more
                - increase the temperatures!
                - uncheck the beat hint button (or reduce the beat unmask duration)
            - if you want the generated audio to sound like the original, but with a different beat structure:
                - uncheck the beat hint button
                - decrease the periodic unmasking to anywhere from 2 to 20
                - slightly decrease the random intensity, to like .95
            

            """)
        with gr.Column():
            gr.Markdown("""
            ## Mask Hints
            - most of the original audio will be masked and replaced with audio generated by vampnet
            - mask hints are used to guide vampnet to generate audio that sounds like the original
            - the more hints you give, the more the generated audio will sound like the original

            """)


    with gr.Row():
        with gr.Column():
            mode = gr.Radio(
                label="**mode**. note that loop mode requires a prefix and suffix longer than 0",
                choices=["standard",],
                value="standard"
            )
            use_coarse2fine = gr.Checkbox(
                label="use coarse2fine",
                value=True
            )
            num_vamps = gr.Number(
                label="number of vamps. more vamps = longer generated audio",
                value=1,
                precision=0, 
                visible=False
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

            mask_periodic_amt = gr.Slider(
                label="periodic hint  (0.0 means no hint, 2 - lots of hints, 8 - a couple of hints, 16 - occasional hint, 32 - very occasional hint, etc)",
                minimum=0,
                maximum=64, 
                step=1,
                value=9, 
            )
            mask_periodic_width = gr.Slider(
                label="periodic hint width (steps, 1 step ~= 10milliseconds",
                minimum=1,
                maximum=100,
                step=1,
                value=1,
            )
            mask_periodic_dropout = gr.Slider(
                label="periodic hint dropout (0.0 means no dropout, 1.0 means all dropout)",
                minimum=0.0,
                maximum=1.0,
                value=0.0,
            )
            mask_periodic_width_dropout = gr.Slider(    
                label="periodic hint width dropout (0.0 means no dropout, 1.0 means all dropout)",
                minimum=0.0,
                maximum=1.0,
                value=0.0,
            )

            rand_mask_intensity = gr.Slider(
                label="random mask intensity. (If this is less than 1, scatters tiny hints throughout the audio, should be between 0.9 and 1.0)",
                minimum=0.8,
                maximum=1.0,
                value=1.0
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

            use_beats = gr.Checkbox(
                label="use beat hints (helps the output stick to the beat structure of the input)",
                value=False
            )

            snap_to_beats = gr.Checkbox(
                label="trim to beat markers (uncheck if the output audio is too short.)",
                value=True
            )

            num_steps = gr.Slider(
                label="number of steps (should normally be between 12 and 36)",
                minimum=1,
                maximum=128,
                step=1,
                value=36
            )

            vamp_button = gr.Button("vamp!!!")

            output_audio = gr.Audio(
                label="output audio",
                interactive=False,
                type="filepath"
            )


            # gr.Markdown("**NOTE**: for loop mode, both prefix and suffix must be greater than 0.")
            # compute_mask_button = gr.Button("compute mask")
            # mask_output = gr.Audio(
            #     label="masked audio",
            #     interactive=False,
            #     visible=False
            # )
            # mask_output_viz = gr.Video(
            #     label="masked audio",
            #     interactive=False
            # )
        
        with gr.Column():
            with gr.Accordion(label="beat unmask (how much time around the beat should be hinted?)"):
                
                beat_unmask_dur = gr.Slider(
                    label="duration", 
                    minimum=0.0,
                    maximum=3.0,
                    value=0.07
                )
                beat_unmask_drop = gr.Slider(
                    label="dropout (within beat)", 
                    minimum=0.0,
                    maximum=1.0,
                    value=0.0
                )

                with gr.Accordion("downbeat settings", open=False):
                    mask_dwn_chk = gr.Checkbox(
                        label="hint downbeats",
                        value=True
                    )
                    dwn_factor = gr.Slider(
                        label="downbeat downsample factor (hint only every Nth downbeat)",
                        value=0, 
                        minimum=0,
                        maximum=16, 
                        step=1
                    )
                with gr.Accordion("upbeat settings", open=False):
                    mask_up_chk = gr.Checkbox(
                        label="hint upbeats",
                        value=True
                    )
                    up_factor = gr.Slider(
                        label="upbeat downsample factor (hint only every Nth upbeat)",
                        value=0,
                        minimum=0,
                        maximum=16,
                        step=1
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
            

            thank_you = gr.Markdown("")

            
    # connect widgets
    vamp_button.click(
        fn=vamp,
        inputs=[input_audio, init_temp,final_temp,
            prefix_s, suffix_s, rand_mask_intensity, 
            mask_periodic_amt, beat_unmask_dur, 
            mask_dwn_chk, dwn_factor, 
            mask_up_chk, up_factor, 
            num_vamps, mode, use_beats, num_steps, snap_to_beats, 
            beat_unmask_drop, mask_periodic_width, 
            mask_periodic_dropout, mask_periodic_width_dropout, 
            n_conditioning_codebooks, use_coarse2fine, stretch_factor
        ],
        outputs=[output_audio, audio_mask], 
        api_name="vamp"
    )

    save_button.click(
        fn=save_vamp,
        inputs=[
            input_audio, init_temp, final_temp,
            prefix_s, suffix_s, rand_mask_intensity,
            mask_periodic_amt, beat_unmask_dur,
            mask_dwn_chk, dwn_factor,
            mask_up_chk, up_factor,
            num_vamps, mode,
            output_audio,
            notes_text, use_beats, num_steps, snap_to_beats, 
            beat_unmask_drop, mask_periodic_width, 
            mask_periodic_dropout, mask_periodic_width_dropout, 
            n_conditioning_codebooks, use_coarse2fine, stretch_factor
        ],
        outputs=[thank_you, download_file]
    )

    ez_vamp_button = gr.Button("ez vamp")
    ez_vamp_button.click(
        fn=ez_vamp,
        inputs=[input_audio, init_temp, final_temp, mask_periodic_amt,
                mask_periodic_width, num_steps, stretch_factor ],
        outputs=[output_audio],
        api_name="ez_vamp"
    )

demo.launch(share=True, enable_queue=False, debug=True)
