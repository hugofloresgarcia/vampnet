# VampNet

This repository contains recipes for training generative music models on top of the Lyrebird Audio Codec.

# Setting up

## Install LAC

install AudioTools

```bash
git clone https://github.com/hugofloresgarcia/audiotools.git
pip install -e ./audiotools
```

install the LAC library. 

```bash
git clone https://github.com/hugofloresgarcia/lac.git
pip install -e ./lac
```

install VampNet

```bash
git clone https://github.com/hugofloresgarcia/vampnet2.git
pip install -e ./vampnet2
```

## A note on Argbind
This repository relies on [argbind](https://github.com/pseeth/argbind) to manage CLIs and config files. 
Config files are stored in the `conf/` folder. 

# How the code is structured

This code was written fast to meet a publication deadline, so it can be messy and redundant at times. Currently working on cleaning it up. 

# Usage

## Staging a Run

Staging a run makes a copy of all the git-tracked files in the codebase and saves them to a folder for reproducibility. You can then run the training script from the staged folder. 

coming soon

## Training a model

```bash
python scripts/exp/train.py --args.load conf/vampnet.yml --save_path /path/to/checkpoints
```

## Fine-tuning
To fine-tune a model, see the configuration files under `conf/lora/`. 
You just need to provide a list of audio files // folders to fine-tune on, then launch the training job as usual.
```bash
python scripts/exp/train.py --args.load conf/lora/birds.yml --save_path /path/to/checkpoints
```

## Launching the Gradio Interface
```bash
python demo.py --args.load conf/interface/spotdl.yml --Interface.device cuda
```