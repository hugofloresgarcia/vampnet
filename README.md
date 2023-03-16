# Lyrebird VampNet

This repository contains recipes for training generative music models on top of the Lyrebird Audio Codec. 

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

### Useful commands

#### Cleaning up after a run

Sometimes DDP runs fail to clear themselves out of the machine. To fix this, run

```bash
cleanup
```
