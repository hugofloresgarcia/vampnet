from pathlib import Path
from typing import Tuple
import yaml

import numpy as np
import audiotools as at
import argbind

import gradio as gr
from vampnet.interface import Interface

conf = yaml.safe_load(Path("conf/interface-jazzpop-exp.yml").read_text())

Interface = argbind.bind(Interface)
AudioLoader = argbind.bind(at.data.datasets.AudioLoader)
with argbind.scope(conf):
    interface = Interface()
    loader = AudioLoader()

dataset = at.data.datasets.AudioDataset(
    loader,
    sample_rate=interface.codec.sample_rate,
    duration=interface.coarse.chunk_size_s,
    n_examples=5000,
    without_replacement=True,
)


def load_audio(file):
    print(file)
    filepath = file.name
    sig = at.AudioSignal.salient_excerpt(
        filepath, 
        duration=interface.coarse.chunk_size_s
    )
    sig = interface.preprocess(sig)

    audio = sig.samples.numpy()[0]
    sr = sig.sample_rate
    return sr, audio.T

def load_random_audio():
    index = np.random.randint(0, len(dataset))
    sig = dataset[index]["signal"]
    sig = interface.preprocess(sig)

    audio = sig.samples.numpy()[0]
    sr = sig.sample_rate
    return sr, audio.T

def mask_audio(
        prefix_s, suffix_s, rand_mask_intensity, 
        mask_periodic_amt, beat_unmask_dur, 
        mask_dwn_chk, dwn_factor, 
        mask_up_chk, up_factor
    ):
    pass

def vamp(
    input_audio, prefix_s, suffix_s, rand_mask_intensity,
    mask_periodic_amt, beat_unmask_dur,
    mask_dwn_chk, dwn_factor,
    mask_up_chk, up_factor
):
    print(input_audio)


with gr.Blocks() as demo:

    gr.Markdown('# Vampnet')
    
    with gr.Row():
        # input audio
        with gr.Column():
            gr.Markdown("## Input Audio")

            manual_audio_upload = gr.File(
                label=f"upload some audio (will be randomly trimmed to max of {interface.coarse.chunk_size_s:.2f}s)",
                file_types=["audio"]
            )
            load_random_audio_button = gr.Button("or load random audio")

            input_audio = gr.Audio(
                label="input audio",
                interactive=False, 
            )
            input_audio_viz = gr.HTML(
                label="input audio",
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
            gr.Markdown("## Mask Settings")
            prefix_s = gr.Slider(
                label="prefix length (seconds)",
                minimum=0.0,
                maximum=10.0,
                value=0.0
            )
            suffix_s = gr.Slider(
                label="suffix length (seconds)",
                minimum=0.0,
                maximum=10.0,
                value=0.0
            )

            rand_mask_intensity = gr.Slider(
                label="random mask intensity (lower means more freedom)",
                minimum=0.0,
                maximum=1.0,
                value=1.0
            )

            mask_periodic_amt = gr.Slider(
                label="periodic unmasking factor (higher means more freedom)",
                minimum=0,
                maximum=32, 
                step=1,
                value=2, 
            )
            compute_mask_button = gr.Button("compute mask")
            mask_output = gr.Audio(
                label="masked audio",
                interactive=False,
                visible=False
            )
            mask_output_viz = gr.Video(
                label="masked audio",
                interactive=False
            )
        
        with gr.Column():
            gr.Markdown("## Beat Unmasking")
            with gr.Accordion(label="beat unmask"):
                beat_unmask_dur = gr.Slider(
                    label="duration", 
                    minimum=0.0,
                    maximum=3.0,
                    value=0.1
                )
                with gr.Accordion("downbeat settings"):
                    mask_dwn_chk = gr.Checkbox(
                        label="unmask downbeats",
                        value=True
                    )
                    dwn_factor = gr.Slider(
                        label="downbeat downsample factor (unmask every Nth downbeat)",
                        value=1, 
                        minimum=1,
                        maximum=16, 
                        step=1
                    )
                with gr.Accordion("upbeat settings"):
                    mask_up_chk = gr.Checkbox(
                        label="unmask upbeats",
                        value=True
                    )
                    up_factor = gr.Slider(
                        label="upbeat downsample factor (unmask every Nth upbeat)",
                        value=1,
                        minimum=1,
                        maximum=16,
                        step=1
                    )
            
    # process and output
    with gr.Row():
        with gr.Column():
            vamp_button = gr.Button("vamp")

            output_audio = gr.Audio(
                label="output audio",
                interactive=False,
                visible=False
            )
            output_audio_viz = gr.Video(
                label="output audio",
                interactive=False
            )

    # connect widgets
    compute_mask_button.click(
        fn=mask_audio,
        inputs=[
            prefix_s, suffix_s, rand_mask_intensity, 
            mask_periodic_amt, beat_unmask_dur, 
            mask_dwn_chk, dwn_factor, 
            mask_up_chk, up_factor
        ],
        outputs=[mask_output, mask_output_viz]
    )

    # connect widgets
    vamp_button.click(
        fn=vamp,
        inputs=[input_audio,
            prefix_s, suffix_s, rand_mask_intensity, 
            mask_periodic_amt, beat_unmask_dur, 
            mask_dwn_chk, dwn_factor, 
            mask_up_chk, up_factor
        ],
        outputs=[output_audio, output_audio_viz]
    )


demo.launch(share=True)