from pathlib import Path
import gradio as gr
import torch
from PIL import Image

from soundmaterial.listen import to_output

from vampnet.interface import Interface
from vampnet.serve import create_param_manager
from vampnet.train import VampNetTrainer
from vampnet.mask import apply_mask
import vampnet.dsp.signal as sn 
from vampnet.util import Timer


CHECKPOINT = "hugggof/vampnetv2-tria-d774-l8-h8-mode-vampnet_rms-hchroma-36c-top3-latest"
device = "cuda" if torch.cuda.is_available() else "cpu"

pm = create_param_manager()


def setup_from_checkpoint(ckpt: str):
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

eiface = setup_from_checkpoint(CHECKPOINT)

def get_param(data, key: str):
    return data[input_widgets[key]]

def process(data):
    # breakpoint()
    insig = sn.Signal(
        torch.from_numpy(data[input_audio][1]).unsqueeze(0), 
        sr=data[input_audio][0]
    )
    inldns = sn.loudness(insig)
    insig = eiface.preprocess(insig)

    # load the sample (if any)
    if data[sample_audio] is not None:
        splsig = sn.Signal(
            data[sample_audio][1], 
            sr=data[sample_audio][0]
        )
        sig_spl = eiface.preprocess(sig_spl)

    # extract onsets, for our onset mask
    onset_idxs = sn.onsets(insig, hop_length=eiface.codec.hop_length)

    # extract controls and build a mask for them
    ctrls = eiface.controller.extract(sig)
    ctrl_masks = {}
    ctrl_masks["rms"] = eiface.rms_mask(
        ctrls["rms"], onset_idxs=onset_idxs, 
        periodic_prompt=get_param(data, "controls_periodic_prompt"), 
        drop_amt=0.3
    )
    ctrl_masks["hchroma-36c-top3"] = torch.zeros_like(ctrl_masks["rms"])
    # ctrl_masks["hchroma-36c-top3"] = ctrl_masks["rms"]

    # encode the signal
    codes = eiface.encode(insig.wav)
    print(f"encoded to codes of shape {codes.shape}")

    # make a mask for the codes
    mask = eiface.build_codes_mask(codes, 
        periodic_prompt=get_param(data, "codes_periodic_prompt"), 
        upper_codebook_mask=get_param(data, "codes_upper_codebook_mask")
    )

    if data[sample_audio] is not None:
        # encode the sample
        codes_spl = eiface.encode(sig_spl.wav)
        print(f"encoded to codes of shape {codes_spl.shape}")

        # add sample to bundle
        codes, mask, ctrls, ctrl_masks = eiface.add_sample(
            spl_codes=codes_spl, codes=codes, 
            cmask=mask, ctrls=ctrls, ctrl_masks=ctrl_masks
        )

    # apply the mask
    mcodes = apply_mask(codes, mask, eiface.vn.mask_token)

    # visualize the bundle
    outvizpath = eiface.visualize(
        sig=insig, 
        codes=mcodes, mask=mask, 
        ctrls=ctrls, ctrl_masks=ctrl_masks
    )
    outviz = Image.open(outvizpath)


    # generate!
    with torch.autocast(device,  dtype=torch.bfloat16):
        gcodes = vn.generate(
            codes=mcodes,
            temperature=get_param(data, "temperature"),
            cfg_scale=5.0,
            mask_temperature=get_param(data, "mask_temperature"),
            typical_filtering=True,
            typical_mass=get_param(data, "typical_mass"),
            ctrls=ctrls,
            ctrl_masks=ctrl_masks,
            typical_min_tokens=128,
            sampling_steps=24 if eiface.vn.mode == "vampnet" else [16, 8, 4, 4],
            causal_weight=0.0,
            debug=False
        )

    # write the generated signal
    generated_wav = eiface.decode(gcodes)

    return to_output(sn.Signal(generated_wav, insig.sr)), outviz


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            # add an input audio widget
            input_audio = gr.Audio(label="Input audio", name="input_audio")
            sample_audio = gr.Audio(label="Sample audio", name="sample_audio")

        with gr.Column():
            input_widgets = {}
            for name, param in pm.asdict().items():
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
            process_button = gr.Button(label="vamp", name="process")

            # add an output image widget
            output_img = gr.Image(label="output viz", type="pil")

            # add an output audio widget
            output_audio = gr.Audio(label="output audio",)

            process_button.click(
                process, 
                inputs={input_audio, sample_audio} | set(input_widgets.values()), 
                outputs=[output_audio, output_img]
            )


demo.launch()                

