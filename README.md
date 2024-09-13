---
title: salad bowl (vampnet)
emoji: 🥗
colorFrom: yellow
colorTo: green
sdk: gradio
sdk_version: 4.37.2
python_version: 3.9.17
app_file: app.py
pinned: false
license: cc-by-nc-4.0
---

# VampNet

This repository contains recipes for training generative music models on top of the Descript Audio Codec.

# Setting up

**Requires Python 3.9**. 

you'll need a Python 3.9 environment to run VampNet. This is due to a [known issue with madmom](https://github.com/hugofloresgarcia/vampnet/issues/15). 

(for example, using conda)
```bash
conda create -n vampnet python=3.9
conda activate vampnet
```

install VampNet

```bash
git clone https://github.com/hugofloresgarcia/vampnet.git
pip install -e ./vampnet
```

# Usage

quick start!
```python
import random
import vampnet
import audiotools as at

# load the default vampnet model
interface = vampnet.interface.Interface.default()

# list available finetuned models
finetuned_model_choices = interface.available_models()
print(f"available finetuned models: {finetuned_model_choices}")

# pick a random finetuned model
model_choice = random.choice(finetuned_model_choices)
print(f"choosing model: {model_choice}")

# load a finetuned model
interface.load_finetuned(model_choice)

# load an example audio file
signal = at.AudioSignal("assets/example.wav")

# get the tokens for the audio
codes = interface.encode(signal)

# build a mask for the audio
mask = interface.build_mask(
    codes, signal,
    periodic_prompt=7, 
    upper_codebook_mask=3,
)

# generate the output tokens
output_tokens = interface.vamp(
    codes, mask, return_mask=False,
    temperature=1.0, 
    typical_filtering=True, 
)

# convert them to a signal
output_signal = interface.decode(output_tokens)

# save the output signal
output_signal.write("scratch/output.wav")
```


## Launching the Gradio Interface
You can launch a gradio UI to play with vampnet. 

```bash
python app.py --args.load conf/interface.yml --Interface.device cuda
```

# Training / Fine-tuning 

## Training a model

To train a model, run the following script: 

```bash
python scripts/exp/train.py --args.load conf/vampnet.yml --save_path /path/to/checkpoints
```

for multi-gpu training, use torchrun:

```bash
torchrun --nproc_per_node gpu scripts/exp/train.py --args.load conf/vampnet.yml --save_path path/to/ckpt
```

You can edit `conf/vampnet.yml` to change the dataset paths or any training hyperparameters. 

For coarse2fine models, you can use `conf/c2f.yml` as a starting configuration. 

See `python scripts/exp/train.py -h` for a list of options.

## Debugging training

To debug training, it's easier to debug with 1 gpu and 0 workers

```bash
CUDA_VISIBLE_DEVICES=0 python -m pdb scripts/exp/train.py --args.load conf/vampnet.yml --save_path /path/to/checkpoints --num_workers 0
```

## Fine-tuning
To fine-tune a model, use the script in `scripts/exp/fine_tune.py` to generate 3 configuration files: `c2f.yml`, `coarse.yml`, and `interface.yml`. 
The first two are used to fine-tune the coarse and fine models, respectively. The last one is used to launch the gradio interface.

```bash
python scripts/exp/fine_tune.py "/path/to/audio1.mp3 /path/to/audio2/ /path/to/audio3.wav" <fine_tune_name>
```

This will create a folder under `conf/<fine_tune_name>/` with the 3 configuration files.

The save_paths will be set to `runs/<fine_tune_name>/coarse` and `runs/<fine_tune_name>/c2f`. 

launch the coarse job: 
```bash
python scripts/exp/train.py --args.load conf/generated/<fine_tune_name>/coarse.yml 
```

this will save the coarse model to `runs/<fine_tune_name>/coarse/ckpt/best/`.

launch the c2f job: 
```bash
python  scripts/exp/train.py --args.load conf/generated/<fine_tune_name>/c2f.yml 
```

## A note on argbind
This repository relies on [argbind](https://github.com/pseeth/argbind) to manage CLIs and config files. 
Config files are stored in the `conf/` folder. 

### Take a look at the pretrained models
All the pretrained models (trained by hugo) are stored here: https://huggingface.co/hugggof/vampnet 

### Licensing for Pretrained Models: 
The weights for the models are licensed [`CC BY-NC-SA 4.0`](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.ml). Likewise, any VampNet models fine-tuned on the pretrained models are also licensed [`CC BY-NC-SA 4.0`](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.ml).

Download the pretrained models from [this link](https://zenodo.org/record/8136629). Then, extract the models to the `models/` folder. 




