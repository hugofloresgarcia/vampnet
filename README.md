# VampNet

This repository contains recipes for training generative music models on top of the Descript Audio Codec.

## try `unloop`
you can try vampnet in a co-creative looper called unloop. see this link: https://github.com/hugofloresgarcia/unloop

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
git clone --recurse-submodules https://github.com/hugofloresgarcia/vampnet.git 
pip install -e ./vampnet
```

## A note on argbind
This repository relies on [argbind](https://github.com/pseeth/argbind) to manage CLIs and config files.
Config files are stored in the `conf/` folder.

## Getting the Pretrained Models

### Licensing for Pretrained Models: 
The weights for the models are licensed [`CC BY-NC-SA 4.0`](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.ml). Likewise, any VampNet models fine-tuned on the pretrained models are also licensed [`CC BY-NC-SA 4.0`](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.ml).

Download the pretrained models from [this link](https://zenodo.org/record/8136629). Then, extract the models to the `models/` folder. 


# Usage

## Launching the Gradio Interface
You can launch a gradio UI to play with vampnet. 

```bash
python app.py --args.load conf/interface.yml --Interface.device cuda
```

# Preprocessing

### create a csv file of audio files
First, create a csv file to index a folder of audio files to use for training. 
```bash
python scripts/pre/create_dataset.py --audio_folder path/to/audio --output_file path/to/metadata.csv
```

### compute dac tokens for your dataset
Now, you can preprocess all the files in that csv with the Descript Audio Codec, to compress the dataset into dac tokens for training. 
```bash
python scripts/pre/condition_workers.py --input_csv /path/to/metadata.csv --output_folder /path/to/codec/files --conditioner_name "dac"
```

### train/val/test split
Finally, you can create a random train/val/test split for the dataset.
```bash
python scripts/pre/split.py --input_csv /path/to/metadata.csv --test_size 0.1 --val_size 0.1 --seed 123
```

## extras

### Inspect the dataset audio
If you want to get information about your dataset such as total duration, statistics on sample_rate, num_channels, etc. as well as get a random sample of the files in your dataset:
```bash
python scripts/pre/inspect_audio.py --input_csv /path/to/metadata.csv --output_dir /path/to/artifacts/folder --sample_files 100
```

# Training / Fine-tuning 

## download a DAC model to use as a tokenizer
```bash
python -m dac download
```
this will download a `.pth` file to `~/.cache/descript/dac/` that you can copy to `models/vampnet/codec.pth`. 

## Training a model

You can edit `conf/base.yml` to change the csv paths for your audio data, or any training hyperparameters. 

See `python scripts/exp/train.py -h` for a list of options.

## Fine-tuning
*todo*
