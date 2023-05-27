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

## A note on argbind
This repository relies on [argbind](https://github.com/pseeth/argbind) to manage CLIs and config files. 
Config files are stored in the `conf/` folder. 

## Getting the Pretrained Models

Download the pretrained models from [this link](). Then, extract the models to the `models/` folder.

# How the code is structured

This code was written fast to meet a publication deadline, so it can be messy and redundant at times. Currently working on cleaning it up. 

```
├── conf         <- (conf files for training, finetuning, etc)
├── demo.py      <- (gradio UI for playing with vampnet)
├── env          <- (environment variables)
│   └── env.sh
├── models       <- (extract pretrained models)
│   ├── spotdl
│   │   ├── c2f.pth     <- (coarse2fine checkpoint)
│   │   ├── coarse.pth  <- (coarse checkpoint)
│   │   └── codec.pth    <- (codec checkpoint)
│   └── wavebeat.pth
├── README.md
├── scripts
│   ├── exp
│   │   ├── eval.py       <- (eval script)
│   │   └── train.py       <- (training/finetuning script)
│   └── utils
├── vampnet
│   ├── beats.py         <- (beat tracking logic)
│   ├── __init__.py
│   ├── interface.py     <- (high-level programmatic interface)
│   ├── mask.py
│   ├── modules
│   │   ├── activations.py 
│   │   ├── __init__.py
│   │   ├── layers.py
│   │   └── transformer.py  <- (architecture + sampling code)
│   ├── scheduler.py      
│   └── util.py
```

# Usage

First, you'll want to set up your environment
```bash
source ./env/env.sh
```

## Staging a Run

Staging a run makes a copy of all the git-tracked files in the codebase and saves them to a folder for reproducibility. You can then run the training script from the staged folder. 

```
stage --name my_run --run_dir /path/to/staging/folder
```

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