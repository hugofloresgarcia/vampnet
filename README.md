# Lyrebird Wav2Wav

This repository contains recipes for training Wav2Wav models.

## Install hooks

First install the pre-commit util:

https://pre-commit.com/#install

    pip install pre-commit  # with pip
    brew install pre-commit  # on Mac

Then install the git hooks

    pre-commit install
    # check .pre-commit-config.yaml for details of hooks

Upon `git commit`, the pre-commit hooks will be run automatically on the stage files (i.e. added by `git add`)

**N.B. By default, pre-commit checks only run on staged files**

If you need to run it on all files:

    pre-commit run --all-files

## Usage & model zoo

To download the model, one must be authenticated to the `lyrebird-research` project on Google Cloud.
To see all available models, run

```bash
python -m wav2wav.list_models
```

which outputs something like this:

```
gs://research-models/wav2wav
└── prod
    └── v3
        └── ckpt
            ├── best
            │   └── generator
            │       ├── ❌ model.onnx
            │       ├── ❌ nvidia_geforce_rtx_2080_ti_11_7.trt
            │       ├── ✅ package.pth
            │       ├── ❌ tesla_t4_11_7.trt
            │       └── ✅ weights.pth
            └── latest
                └── generator
                    ├── ❌ package.pth
                    └── ❌ weights.pth
    └── v2
        ...
└── dev
    ...
```

This will show all the models that are available on GCP. Models that are available locally are marked with a ✅, while those not available locally
are marked with ❌. `.onnx` indicates a model that must be run with
the `ONNX` runtime, while `.trt` indicate models that have been optimized
with TensorRT. Note that TensorRT models are specific to GPU and CUDA
runtime, and their file names indicate what to use to run them.

`package.pth` is a version of the model that is saved using `torch.package`,
and contains a copy of the model code within it, which allow it to work
even if the model code in `wav2wav/modules/generator.py` changes. `weights.pth`
contains the model weights, and the code must match the code used
to create the model.

To use a model from this list, simply write its path and give it to the `enhance` script,
like so:

```
python -m wav2wav.interface \
  [input_path]
  --model_path=prod/v3/ckpt/best/generator/weights.pth
  --output_path [output_path]
```

Models are downloaded to the location set by the environment variable `MODEL_LOCAL_PATH`, and defaults to `~/.wav2wav/models`. Similarly,
The model bucket is determined by `MODEL_GCS_PATH` and defaults to
`gs://research-models/wav2wav/`.

## Development
### Setting everything up

Run the setup script to set up your environment via:

```bash
python env/setup.py
```

The setup script does not require any dependencies beyond just Python.
Once run, follow the instructions it prints out to create your
environment file, which will be at `env/env.sh`.

Note that if this is a new machine, and
the data is not downloaded somewhere on it already, it will ask you
for a directory to download the data to.

For Github setup, if you don't have a .netrc token, create one by going to your Github profile -> Developer settings -> Personal access tokens -> Generate new token. Copy the token and [keep it secret, keep it safe](https://www.youtube.com/watch?v=iThtELZvfPs).

When complete, run:

```bash
source env/env.sh
```

Now build and launch the Docker containers:

```bash
docker compose up -d
```

This builds and runs a Jupyter notebook and Tensorboard
in the background, which points to your `TENSORBOARD_PATH`
env. variable.

Now, launch your development environment via:

```bash
docker compose run dev
```

To tear down your development environment, just do

```bash
docker compose down
```

### Downloading data and pre-processing
Next, from within the Docker environment (or an appropriately configured Conda environment with environment variables set as above), do the following:

```
python -m wav2wav.preprocess.download
```

This will download all the necessary data, which are referenced by
the CSV files in `conf/audio/*`. These CSVs were generated via
`python -m wav2wav.preprocess.organize`.

### Launching an experiment

Experiments are first _staged_ by running the `stage` command (which corresponds to the script `scripts/exp/stage.py`).

`stage` creates a directory with a copy of all of the Git-tracked files in the root repository.`stage` launches a shell into said directory, so all commands are run on the
copy of the original repository code. This is useful for rewinding to an old experiment
and resuming it, for example. Even if the repository code changes, the snapshot in the experiment directory is unchanged from the original run, so it can be re-used.

Then, the experiment can be run via:

```bash
torchrun --nproc_per_node gpu \
  scripts/exp/train.py \
  --args.load=conf/args.yml \
```

The full settings are in [conf/daps/train.yml](conf/daps/train.yml).

### Evaluating an experiment

There are two ways to evaluate an experiment: quantitative and qualitative.
For the first, we can use the `scripts/exp/evaluate.py` script. This script evaluates the model over the `val_data` and `test_data`, defined in your
`train` script, and takes as input an experiment directory. The metrics
computed by this script are saved to the same folder.

The other way is via a preference test. Let's say we want to compare
the v3 prod model against the v2 prod model. to do this, we use the
`scripts/exp/qa.py` script. This script creates a zip file containing all
the samples and an HTML page for easy viewing. It also creates a Papaya
preference test. Use it like this:

```bash
WAV2WAV_MODELS=a,b python scripts/exp/qa.py \
  --a/model_path prod/v3/ckpt/best/generator/package.pth \
  --b/model_path prod/v2/ckpt/best/generator/package.pth \
  --a/name "v3" --b/name "v2" \
  --device cuda:0 \
  --n_samples 20 \
  --zip_path "samples/out.zip"
```

### Useful commands

#### Monitoring the machine

There's a useful `tmux` workspace that you can launch via:

```bash
tmuxp load ./workspace.yml
```

which will have a split pane with a shell to launch commands on the left,
and GPU monitoring, `htop`, and a script that watches for changes in your
directory on the right, in three split panes.

#### Cleaning up after a run

Sometimes DDP runs fail to clear themselves out of the machine. To fix this, run

```bash
cleanup
```

### Deploying a new model to production

Okay, so you ran a model and it seems promising and you want to upload it
to GCS so it can be QA'd fully, and then shipped. First, upload
your experiment to the `dev` bucket on GCS via:

```bash
gsutil cp -r /path/to/{exp_name} gs://research-models/wav2wav/dev/{exp_name}
```

Once uploaded, QA can access the models by specifying
`model_path=dev/{exp_name}/ckpt/{best,latest}/generator/package.pth` when using the
`wav2wav.interface.enhance` function. If it passes QA, and is scheduled to
ship to production, then next we have to generate the TensorRT model file,
which requires us to have a machine that matches that of a production machine.

There is a script that automates this procedure, that does not require any
fiddling from our end. Navigate to the repository root and run:

```
python scripts/utils/convert_on_gcp.py dev/{exp_name}/ckpt/{best,latest}//generator/weights.pth
```

This will provision the machine, download the relevant model from GCS, optimize it on
the production GPU with the correct CUDA runtime, and then upload the generated `.trt`
and `.onnx` models back to the bucket.

Finally, copy the model to the `prod` bucket, incrementing the version number by one:

```bash
gsutil cp -r gs://research-models/wav2wav/dev/{exp_name} gs://research-models/wav2wav/prod/v{N}
```

where `N` is the next version (e.g. if v3 is the latest, the new one is v4). Then, update
the model table in [Notion](https://www.notion.so/descript/fc04de4b46e6417eba1d06bdc8de6c75?v=e56db4e6b37c4d9b9eca8d9be15c826a) with the new model.

Once the above is all done, we update the code in two places:

1. In `interface.py`, we update `PROD_MODEL_PATH` to point to the `weights.pth`
   for whichever tag ended up shipping (either `best` or `latest`).
2. In `interface.py`, we update `PROD_TRT_PATH` to point the generated
   TensorRT checkpoint generated by the script above.

After merging to master, a new Docker image will be created, and one can update the relevant lines
in descript-workflows like in this [PR](https://github.com/descriptinc/descript-workflows/pull/477/files).

We have Github action workflows in [.github/workflows/deploy.yml](.github/workflows/deploy.yml) to build and deploy new docker images. Two images are built - one for staging and another for production.
To deploy a new release version, follow the instructions in [this coda doc](https://coda.io/d/Research-Engineering_dOABAWL46p-/Deploying-Services_su1am#_lu7E8).

Coda doc with informations about deploying speech-enhance worker is [here](https://coda.io/d/Research-Engineering_dOABAWL46p-/Deploying-Services_su1am#_lu7E8).

And that's it! Once the new staging is built, you're done.

## Testing

### Profiling and Regression testing

- The [profiling script](tests/profile_inference.py) profiles the `wav2wav.interface.enhance` function.
- NOTE: ALWAYS run the profiler on a T4 GPU. ALWAYS run the profiling in isolation i.e kill all other processes on the GPU. Recommended vm size on GCP is `n1-standard-32` as the stress test of six hours of audio requires ~35GB of system memory.
- To run profiling use the [profiling script](tests/profile_inference.py) via command `python3 -m tests.profile_inference`. Results will be printed after `1` run.
- Use the [test_regression.py](tests/test_regression.py) script to run tests that
  - compare performance stats of current model with known best model
  - test for output deviation from the last model
- Run `git lfs checkout` to checkout input file and model weights required for testing the model.
- To launch these tests, run `python3 -m pytest tests/test_regression.py -v`.
- As a side effect, this will update the `tests/stat.csv` file if the current model performs better than last best known model as per `tests/stat.csv`.
- NOTE: In case of architecture change, purge the weights files : `tests/assets/{quick|slow}.pth` and reference stat file : `tests/assets/baseline.json` file. Running the [test_regression.py](tests/test_regression.py) script in absence of reference stat file, will generate new baseline referece stats as well as append new performance stats to stats file. In the absence of saved weights, new weights are generated and saved on disk. Make sure to commit these files (stat.csv, baseline.json, *.pth) when the model architecture changes.

### Unit tests
Regular unit tests that test functionality such as training resume etc. These are run on CPU. Update them when new features are added.

### Profiling tests
These tests profile the model's resource consumption. They are run on T4 GPU with 32 cores and >35GB memory. Their usage is reported in the above sections.

### Functional tests
These tests detect deviation from known baseline model. A category of these tests ensure that a new pytorch model doesn't deviate from the previous one. Another category ensures that the TensorRT version of the current pytorch model doens't deviate from it. These tests are marked with the marker `output_qa` and can be run via the command line `python3 -m pytest -v -m output_qa`. Some of these tests require a GPU.

### CI tests
- The tests are divided into two categories depending on the platform requirement - CPU tests and GPU tests.
- The CPU tests contains unit tests.
- The GPU tests contain a subset of functional tests. These tests can be run by the command `python3 -m pytest -v -m gpu_ci_test`.
