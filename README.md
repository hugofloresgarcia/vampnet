---
title: salad bowl (vampnet)
emoji: 🥗
colorFrom: yellow
colorTo: green
sdk: gradio
sdk_version: 4.43.0
python_version: 3.9.17
app_file: app.py
pinned: false
license: cc-by-nc-4.0
---

# VampNet

# Table of contents

- [setting up](#setting-up)
- [programmatic usage](#programmatic-usage)
- [launching the web app](#launching-the-web-app)
- [training / fine-tuning](#training--fine-tuning)
  - [training a model](#training-a-model)
  - [debugging training](#debugging-training)
  - [fine-tuning](#fine-tuning)
- [exporting your model](#exporting-your-model)
- [unloop](#unloop)
- [token telephone](#token-telephone)
- [a note on argbind](#a-note-on-argbind)
- [take a look at the pretrained models](#take-a-look-at-the-pretrained-models)
- [licensing for pretrained models](#licensing-for-pretrained-models)

## setting up

python 3.9-3.11 works well. (for example, using conda)
```bash
conda create -n vampnet python=3.9
conda activate vampnet
```

install VampNet

```bash
git clone https://github.com/hugofloresgarcia/vampnet.git
pip install -e ./vampnet
```

## programmatic usage

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


# Launching the Web app
You can launch a gradio UI to play with vampnet. 

```bash
python app.py 
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

# Fine-tuning

To fine-tune a model, use the script in `scripts/exp/fine_tune.py` 

for an audio folder
```bash
python scripts/exp/fine_tune.py /path/to/audio/folder <fine_tune_name>
```

for multiple files
```bash
python scripts/exp/fine_tune.py "/path/to/audio1.mp3 /path/to/audio2/ /path/to/audio3.wav" <fine_tune_name>
```

This creates configuration files for a fine tuning train job. The save_paths will be set to `runs/<fine_tune_name>/coarse` and `runs/<fine_tune_name>/c2f`. 

launch the coarse job: 
```bash
python scripts/exp/train.py --args.load conf/generated/<fine_tune_name>/coarse.yml 
```

this will save the coarse model to `runs/<fine_tune_name>/coarse/ckpt/best/`.

launch the c2f job: 
```bash
python  scripts/exp/train.py --args.load conf/generated/<fine_tune_name>/c2f.yml 
```

# Resuming a Training/Finetuning Job from checkpoint. 

To resume from checkpoint, use the `--resume` flag and the `--save_path` to point to the checkpoint you want to resume from.
```bash
python scripts/exp/train.py --args.load conf/generated/steve/coarse.yml --save_path runs/steve/coarse --resume
```

# Exporting your model

Once your model has been fine-tuned, you can export it to a HuggingFace model. 

In order to use your model in `app.py`, you will need to export it to HuggingFace.

**NOTE**: In order to export, you will need a [huggingface account](https://huggingface.co/). 

<!-- You need to fork the [vampnet models repo](https://huggingface.co/hugggof/vampnet) which stores the default vampnet models.  -->

Now, create a repo on huggingface. You can do this by going to the huggingface website and clicking on the "Create a new model" button.
Copy your repo name. It should look something like `<USERNAME>/vampnet`.

Now, navigate to `models/vampnet` and add your new repo as a remote:

```bash
cd ./models/vampnet && git init && git remote add origin https://huggingface.co/<YOUR_REPO_NAME> && git pull origin main
```
go back

```bash
cd ../../
```

Now, replace the contents of the file named `./DEFAULT_HF_MODEL_REPO` in the root folder with the name of your repo (usually `<USERNAME>/vampnet`). 

Now, log in to huggingface using the command line:
```bash
huggingface-cli login
```

Now, run the following command to export your model:

```bash
python scripts/exp/export.py --name <your_finetuned_model_name> --model latest
```

Once that's done, your model should appear on the list of available models in the gradio interface.
Simply run `python app.py` and select your model from the dropdown list.


# Unloop

Make sure you have Max installed on your laptop!

**NOTE**: To run unloop (with a GPU-powered server), you will need to install the vampnet repo in both your local machine and your GPU server.

## start a vampnet gradio server

First, **on your GPU server**, run the gradio server:
```bash
python app.py --args.load conf/interface.yml --Interface.device cuda
```
This will run a vampnet gradio API on your GPU server. Copy the address. It will be something like `https://127.0.0.1:7860/`. 

**IMPORTANT** Make sure that this gradio port (by default `7860`) is forwarded to your local machine, where you have Max installed. 

## start the unloop gradio client
Now, **on your local machine**, run the unloop gradio client.
```
cd unloop
pip install -r requirements.txt
python client.py --vampnet_url https://127.0.0.1:7860/ # replace with your gradio server address
```
This will start a gradio client that connects to the gradio server running on your GPU server.

## start the unloop Max patch
Now, open the unloop Max patch. It's located at `unloop/max/unloop.maxpat`.

In the tape controls, check the heartbeat (`<3`) to make sure the connection to the local gradio client is working. 

have fun!

# Token Telephone

Instructions forthcoming, but the sauce is in `token_telephone/tt.py`

## A note on argbind
This repository relies on [argbind](https://github.com/pseeth/argbind) to manage CLIs and config files. 
Config files are stored in the `conf/` folder. 

### Take a look at the pretrained models
All the pretrained models (trained by hugo) are stored here: https://huggingface.co/hugggof/vampnet 

### Licensing for Pretrained Models: 
The weights for the models are licensed [`CC BY-NC-SA 4.0`](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.ml). Likewise, any VampNet models fine-tuned on the pretrained models are also licensed [`CC BY-NC-SA 4.0`](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.ml).

Download the pretrained models from [this link](https://zenodo.org/record/8136629). Then, extract the models to the `models/` folder. 




