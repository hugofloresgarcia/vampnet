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
device = "cuda" if torch.cuda.is_available() else "cpu"

pm = create_param_manager()

def setup_from_checkpoint(ckpt: str = DEFAULT_CHECKPOINT):
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
    
    seed = get_param(data, "seed")
    controls_periodic_prompt = get_param(data, "controls_periodic_prompt")
    codes_periodic_prompt = get_param(data, "codes_periodic_prompt")
    upper_codebook_mask = get_param(data, "codes_upper_codebook_mask")
    temperature = get_param(data, "temperature")
    mask_temperature = get_param(data, "mask_temperature")
    typical_mass = get_param(data, "typical_mass")

    timer = Timer()
        
    if data[input_widgets["randomize_seed"]] or seed < 0:
        import time
        seed = time.time_ns() % (2**32-1)

    print(f"using seed {seed}")
    sn.seed(seed)

    # preprocess the input signal
    timer.tick("preprocess")
    insig = sn.to_mono(insig)
    inldns = sn.loudness(insig)
    insig = eiface.preprocess(insig)

    # load the sample (if any)
    if sig_spl is not None:
        sig_spl = sn.to_mono(sig_spl)
        sig_spl = eiface.preprocess(sig_spl)
    timer.tock("preprocess")

    timer.tick("controls")
    # extract controls and build a mask for them
    ctrls = eiface.controller.extract(insig)
    ctrl_masks = {}
    if len(ctrls) > 0:
        # extract onsets, for our onset mask
        onset_idxs = sn.onsets(insig, hop_length=eiface.codec.hop_length)
        ctrl_masks["rms"] = eiface.rms_mask(
            ctrls["rms"], onset_idxs=onset_idxs, 
            periodic_prompt=controls_periodic_prompt, 
            drop_amt=0.3
        )
        # use the rms mask for the other controls
        for k in ctrls.keys():
            if k != "rms":
                ctrl_masks[k] = ctrl_masks["rms"]
                # alternatively, zero it out
                # ctrl_masks[k] = torch.zeros_like(ctrl_masks["rms"])
    timer.tock("controls")

    timer.tick("encode")
    # encode the signal
    codes = eiface.encode(insig.wav)
    timer.tock("encode")

    # make a mask for the codes
    mask = eiface.build_codes_mask(codes, 
        periodic_prompt=codes_periodic_prompt, 
        upper_codebook_mask=upper_codebook_mask
    )

    timer.tick("prefix")
    if sig_spl is not None:
        # encode the sample
        codes_spl = eiface.encode(sig_spl.wav)

        # add sample to bundle
        codes, mask, ctrls, ctrl_masks = eiface.add_sample(
            spl_codes=codes_spl, codes=codes, 
            cmask=mask, ctrls=ctrls, ctrl_masks=ctrl_masks
        )
    timer.tock("prefix")

    # apply the mask
    mcodes = apply_mask(codes, mask, eiface.vn.mask_token)

    # generate!
    timer.tick("generate")
    with torch.autocast(device,  dtype=torch.bfloat16):
        gcodes = eiface.vn.generate(
            codes=mcodes,
            temperature=temperature,
            cfg_scale=5.0,
            mask_temperature=mask_temperature,
            typical_filtering=True,
            typical_mass=typical_mass,
            ctrls=ctrls,
            ctrl_masks=ctrl_masks,
            typical_min_tokens=128,
            sampling_steps=24 if eiface.vn.mode == "vampnet" else [16, 8, 4, 4],
            causal_weight=0.0,
            debug=False
        )
    timer.tock("generate")

    # remove codes
    if sig_spl is not None:
        gcodes = eiface.remove_sample(codes_spl, gcodes)

    timer.tick("decode")
    # write the generated signal
    generated_wav = eiface.decode(gcodes)
    timer.tock("decode")

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
        return to_output(sn.Signal(generated_wav, insig.sr)), outviz, seed
    else:
        return to_output(sn.Signal(generated_wav, insig.sr))


def process_api(data):
    return process(data, return_img=False)

def process_normal(data):
    return process(data, return_img=True)

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


demo.launch()                

