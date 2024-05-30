# vampnet!
music generation via masked acoustic token modeling. 

## take the tour! 
to get started using pretrained vampnet models and fine-tuning your own, 
check out the [tour](/tour.ipynb). 

## install 
<!-- (coming soon)
```bash 
pip install vampnet
``` -->

<!-- for now -->
```bash
git clone --recursive https://github.com/hugofloresgarcia/vampnet
pip install -e ./vampnet
```

### play with vampnet!

try the gradio demo
```bash
python app.py 
```

or use it programatically
```python
import audiotools as at
import vampnet

# load the audio tokenizer
codec = vampnet.load_codec()

# load the default pretrained model
model = vampnet.load_model("hugggof/vampnet:vampnet-base-best")

# put them into an interface
interface = vampnet.Interface(codec, model)

# load an example audio file
signal = at.AudioSignal("assets/example_audio/fountain.mp3")

# get the tokens for the audio
codes = interface.encode(signal)

# build a mask for the audio
mask = interface.build_mask(signal, 
    periodic_prompt=7, 
    upper_codebook_mask=2,
)

# generate the output tokens
output_tokens = interface.vamp(
    vamp, mask, return_mask=False,
    temperature=1.0, 
    typical_filtering=True, 
    top_p=0.8,
    sample_cutoff=1.0, 
)

# convert them to a signal
output_signal = interface.to_signal(output_tokens)

# save the output signal
output_signal.save("scratch/output.wav")
```


## command line usage

### note: creating a custom config
if you want to use vampnet with a custom configuration, you will need a custom config file. 
for an example config, see `config/example.py`

for all available config variables, see `vampnet/config/defaults.py`

### training // fine tuning vampnet 
before we can add any data, we need to create a database. 
```bash 
python -m vampnet.db.init --config config/example.py
```

now, we can add a dataset (in the form of an audio folder) to the db.
```bash 
python -m vampnet.db.create --config config/example.py --audio_folder assets/example_audio/ --dataset_name example
```

before we can train or fine-tune, we need to preprocess our dataset. 
```bash
python -m vampnet.db.preprocess --config config/example.py --dataset example
```

now, we can fine tune a model on our dataset. 
```bash
python -m vampnet.fine_tune --config config/example.py --dataset example 
```

or train it from scratch, though, with the example config, this will require a large dataset (20k ish hours at a minimum). 
```bash
python -m vampnet.train --config config/example.py --dataset example 
```


## using with pyharp

TODO


## programmatic usage

```python
import yapecs

# load vampnet with it's default config
import vampnet

# or alternatively, load from a custom config. 
# REPLACE config/vampnet.py WITH YOUR OWN CONFIGURATION
vampnet = yapecs.compose("vampnet", ["config/vampnet.py"])
```

### pretrained models
view a list of locally available pretrained models, and load a local pretrained model

```python
available_local_models = vampnet.list_local_models()
print(f"available_local_models: {available_local_models}")

# load the local model
model_name = available_local_models[0]
model = vampnet.load_local_model(model_name)
```

view a list of available pretrained models in the HF hub, and load one
```python
available_hub_models = vampnet.list_hub_models()
print(f"available hub models: {available_hub_models}")

# load the hub model
model_id = available_hub_models[0]
model = vampnet.load_hub_model(model_id)
```

### custom configs
you will need a custom config file to configure vampnet. 
for an example config, see `config/example.py`

for all available config variables, see `vampnet/config/defaults.py`

```python
# REPLACE config/vampnet.py WITH YOUR OWN CONFIGURATION
vampnet = yapecs.compose("vampnet", ["config/example.py"])
print("default config: ", vampnet.CONFIG)
print("custom config: ", vampnet_custom.CONFIG)
```



## packaging and uploading to pypi

install twine
```bash
pip install twine
```

build the package
```bash
python setup.py sdist
```

upload to test pypi
```bash
twine upload --repository testpypi dist/*
```

upload to pypi
```bash
twine upload dist/*
```


