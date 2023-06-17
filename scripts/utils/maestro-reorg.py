from pathlib import Path
import json
import os

maestro_path = Path("/media/CHONK/hugo/maestro-v3.0.0")
output_path = Path("/media/CHONK/hugo/maestro-v3.0.0-split")

# split
with open(maestro_path / "maestro-v3.0.0.json") as f:
    maestro = json.load(f)

breakpoint()
train = []
validation = []
test = []
for key, split in maestro["split"].items():
    audio_filename = maestro['audio_filename'][key]
    if split == "train":
        train.append(audio_filename)
    elif split == "test":
        test.append(audio_filename)
    elif split == "validation":
        validation.append(audio_filename)
    else:
        raise ValueError(f"Unknown split {split}")

# symlink all files
for audio_filename in train:
    p = output_path / "train" / audio_filename
    p.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(maestro_path / audio_filename, p)
for audio_filename in validation:
    p = output_path / "validation" / audio_filename
    p.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(maestro_path / audio_filename, p)
for audio_filename in test:
    p = output_path / "test" / audio_filename
    p.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(maestro_path / audio_filename, p)