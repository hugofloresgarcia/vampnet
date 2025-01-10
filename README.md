# vampnet

This repository contains recipes for training generative music models on top of the Descript Audio Codec.

# setup

**Requires Python 3.11**. 

(for example, using conda)
```bash
conda create -n vampnet python=3.11
conda activate vampnet
```

install vampnet and its dependencies
```bash
git clone --recurse-submodules https://github.com/hugofloresgarcia/vampnet.git
cd vampnet
pip install -e .
pip install -e ./x-transformers
pip install -e ./soundmaterial
```

you'll need download to download the dac codec
```bash
python -m vampnet.dac.download
```

# usage


quick start! see `scripts/hello-vampnet.py` for an example of how to programmatically use a pretrained vampnet for inference. 

# training
## setting up a database for training

create a soundmaterial database and add a folder of audio files to it.
```bash
python -m soundmaterial.create sm.db # creates a new database at sm.db
python -m soundmaterial.add sm.db /path/to/audio-1/ my-dataset-1 # adds audio files to db, with a dataset name of my-dataset
python -m soundmaterial.add sm.db /path/to/audio-2/ my-dataset-2 # add a second dataset
```

make a table of 5 second chunks to uniformly sample from them. 
```bash
python -m soundmaterial.chunk sm.db 5.0 # creates a table of 5 second chunks
```

## training a model

To train a model, run the following script: 

```bash
CUDA_VISIBLE_DEVICES=0 python -m vampnet.train --args.load conf/vampnet.yml
```

you can resume from a checkpoint by specifying the `--resume_ckpt` flag. 

vampnet will use as many GPUs as you have in your `CUDA_VISIBLE_DEVICES` environment variable.

You can edit `conf/vampnet.yml` to change the dataset paths, SQL queries or any training hyperparameters. 

## export a pretrained model

first, login to the huggingface model hub
```bash
huggingface-cli login
```

once you have a trained model you like, you can export a pretrained model to the huggingface model hub like this:
```
python scripts/export.py --ckpt runs/.../checkpoints/best.ckpt --hf_repo hugggof/vampnetv2 --version_tag latest
```

## debugging training

To debug training, it's easier to debug with 1 gpu and 0 workers

```bash
CUDA_VISIBLE_DEVICES=0 python -m pdb -m vampnet.train --args.load conf/vampnet.yml --save_path /path/to/checkpoints --num_workers 0
```


## more database tricks

(*optional*) explore your dataset by listening to it
```bash
python -m soundmaterial.listen sm.db
```

(*optional*) or by looking each and every single file
```bash
pip install sqlite_web
sqlite_web sm.db
```

you can train on subsets of the data by modifying the sql query in `conf/vampnet.yml`

for example: 
```yaml
build_datasets.db_path: sm.db
build_datasets.query: "
    SELECT af.path, chunk.offset, chunk.duration, af.duration as total_duration, dataset.name 
    FROM chunk 
    JOIN audio_file as af ON chunk.audio_file_id = af.id 
    JOIN dataset ON af.dataset_id = dataset.id
    WHERE dataset.name IN ('my-dataset-1', 'my-dataset-2')
"
```
NOTE: the SQL query **MUST** return the following columns: `path`, `offset`, `duration`, `total_duration`. see the example query above for reference.

# gloop (generative looper)

to run gloop, the generative looper interface that runs vampnet at its core, you need to do two steps. 

first, start a gloop server. 

**TODO**: needs an ability to select a checkpoint
```bash
python -m vampnet.serve --ckpt hugggof/vampnetv2-mode-vampnet_rms-latest --device mps
```

then, start the gloop puredata patch. 
```bash
/path/to/pd pd/looper.pd
```

## A note on argbind
This repository relies on [argbind](https://github.com/pseeth/argbind) to manage CLIs and config files. 
Config files are stored in the `conf/` folder. 

