from pathlib import Path
import gradio as gr
import torch
from functools import partial
from PIL import Image

import argbind
import numpy as np

from vampnet.interface import Interface
from vampnet.serve import create_param_manager
from vampnet.train import VampNetTrainer
from vampnet.mask import apply_mask
import vampnet.dsp.signal as sn 
from vampnet.util import Timer


DEFAULT_CHECKPOINT = "hugggof/vampnetv2-tria-d1026-l8-h8-mode-vampnet_rms-hchroma-36c-top3-latest"

pm = create_param_manager()

def setup_from_checkpoint(ckpt: str = DEFAULT_CHECKPOINT, 
                          device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    # load a pretrained model bundle
    bundle = VampNetTrainer.from_pretrained(ckpt) 

    # create an interface with our pretrained shizzle
    eiface = Interface(
        codec=bundle.codec,
        vn=bundle.model,
        controller=bundle.controller
    )
    eiface.to(device)
    return eiface

setup_from_checkpoint = argbind.bind(setup_from_checkpoint, without_prefix=True)
args = argbind.parse_args()
with argbind.scope(args):
    eiface = setup_from_checkpoint()

def get_param(data, key: str):
    return data[input_widgets[key]]

def signal_from_gradio(value: tuple):
    wav = torch.from_numpy(value[1]).T.unsqueeze(0)
    if wav.ndim == 2:
        wav = wav.unsqueeze(0)
    sig = sn.Signal(wav, value[0])
    # convert from int16 to float32
    if sig.wav.dtype == torch.int16:
        sig.wav = sig.wav.float() / np.iinfo(np.int16).max
    elif sig.wav.dtype == torch.int32:
        sig.wav = sig.wav.float() / np.iinfo(np.int32).max
    assert sig.wav.dtype == torch.float32, f"expected float32, got {sig.wav.dtype} with min {sig.wav.min()} and max {sig.wav.max()}"
    return sig

def to_output(sig: sn.Signal):
    wave = sn.to_mono(sig).wav[0][0].cpu().numpy()
    return sig.sr, wave * np.iinfo(np.int16).max

def process(data, return_img: bool = True):
    # input params (i'm refactoring this)
    insig = signal_from_gradio(data[input_audio])
    sig_spl = signal_from_gradio(data[sample_audio]) if data[sample_audio] is not None else None

    # if insig is all zeros, return a silent signal
    if insig.wav.sum() == 0:
        print("input signal is all zeros, returning silent signal")
        if return_img:
            return to_output(insig), Image.new("RGB", (1, 1))
        else:
            return to_output(insig)

    randomize_seed = data[input_widgets["randomize_seed"]]
    seed = get_param(data, "seed")
    controls_periodic_prompt = get_param(data, "controls_periodic_prompt")
    controls_drop_amt = get_param(data, "controls_drop_amt")
    codes_periodic_prompt = get_param(data, "codes_periodic_prompt")
    upper_codebook_mask = get_param(data, "codes_upper_codebook_mask")
    temperature = get_param(data, "temperature")
    mask_temperature = get_param(data, "mask_temperature")
    typical_mass = get_param(data, "typical_mass")

    timer = Timer()

    out = eiface.vamp(
        insig=insig,
        sig_spl=sig_spl,
        seed=seed,
        randomize_seed=randomize_seed,
        controls_periodic_prompt=controls_periodic_prompt,
        controls_drop_amt=controls_drop_amt,
        codes_periodic_prompt=codes_periodic_prompt,
        upper_codebook_mask=upper_codebook_mask,
        temperature=temperature,
        mask_temperature=mask_temperature,
        typical_mass=typical_mass,
    )

    outsig = out["sig"]
    mcodes = out["mcodes"]
    mask = out["mask"]
    ctrls = out["ctrls"]
    ctrl_masks = out["ctrl_masks"]

    if return_img:
        # visualize the bundle
        timer.tick("viz")
        outvizpath = eiface.visualize(
            sig=insig, 
            codes=mcodes, mask=mask, 
            ctrls=ctrls, ctrl_masks=ctrl_masks
        )
        outviz = Image.open(outvizpath)
        timer.tock("viz")
        return to_output(outsig), outviz, seed
    else:
        return to_output(outsig)

def process_api(data):
    return process(data, return_img=False)

def process_normal(data):
    return process(data, return_img=True)


def process_preview(data):
    return eiface.preview_input(
        signal_from_gradio(data[input_audio]),
        controls_periodic_prompt=get_param(data, "controls_periodic_prompt"),
        controls_drop_amt=get_param(data, "controls_drop_amt"),
        codes_periodic_prompt=get_param(data, "codes_periodic_prompt"),
        upper_codebook_mask=get_param(data, "codes_upper_codebook_mask"),
    )

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            # add an input audio widget
            input_audio = gr.Audio(label="input_audio")
            sample_audio = gr.Audio(label="sample_audio")

        with gr.Column():
            input_widgets = {}
            for name, param in pm.asdict().items():
                if name == "seed":
                    input_widgets["randomize_seed"] = gr.Checkbox(label="randomize seed", value=True)

                if param.range is not None:
                    # if we have param range, make it a slider
                    input_widgets[name] = gr.Slider(
                        minimum=param.range[0],
                        maximum=param.range[1],
                        step=param.step,
                        label=param.name,
                        value=param.value,
                    )
                else:
                    # otherwise, it's a number
                    input_widgets[name] = gr.Number(
                        label=param.name,
                        value=param.value,
                    )

        with gr.Column():
            process_button = gr.Button(value="vamp",)
            api_process_button = gr.Button(value="api-vamp",)
            preview_button = gr.Button(value="preview-inputs",)

            # add an output audio widget
            output_audio = gr.Audio(label="output audio",)

            # add an output image widget
            output_img = gr.Image(label="output viz", type="pil")

            process_button.click(
                process_normal, 
                inputs={input_audio, sample_audio} | set(input_widgets.values()), 
                outputs=[output_audio, output_img, input_widgets["seed"]], 
            )

            api_process_button.click(
                process_api, 
                inputs={input_audio, sample_audio} | set(input_widgets.values()), 
                outputs=[output_audio], api_name="api-vamp"
            )

            preview_button.click(
                process_preview, 
                inputs={input_audio, sample_audio} | set(input_widgets.values()), 
                outputs=[output_img], api_name="preview-inputs"
            )
            


demo.launch(share=True, debug=True)                
