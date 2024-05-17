import random
from pathlib import Path
import argbind
import audiotools as at
from audiotools import AudioSignal
from multiprocessing import cpu_count 
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import shutil
import pandas as pd
import torch
import numpy as np
import tqdm

from vampnet.condition import REGISTRY, ConditionFeatures, ChromaStemConditioner, YamnetConditioner
from torch.utils.data import Dataset, DataLoader

class AudioDataset(Dataset):
    def __init__(self, audio_files, audio_root, output_folder, conditioner_name, file_ext=".emb"):
        self.audio_root = audio_root
        self.output_folder = output_folder
        self.conditioner_name = conditioner_name
        self.file_ext = file_ext 
        
        # Filter audio files based on existence of corresponding processed output
        self.audio_files = [af for af in audio_files if not self._output_exists(af)]
        print(f"Found {len(self.audio_files)} audio files to process")

    def _output_exists(self, audio_file):
        output_path = Path(self.output_folder) / audio_file.with_suffix(self.file_ext)
        return output_path.exists()

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, index):
        audio_file = self.audio_files[index]
        try:
            sig = AudioSignal(self.audio_root / audio_file)
            sig.path_to_file = audio_file
            return sig
        except Exception as e:
            print(f"Error reading {audio_file}. skipping")
            print(e)
            return None

def receptive_field(model):
    """
    Computes the size, stride and padding of the given model's receptive
    field under the assumption that all its Conv1d and TransposeConv1d
    layers are applied in sequence.
    """
    total_size, total_stride, total_padding = 1, 1, 0
    for layer in model.modules():
        if isinstance(layer, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
            layer_size = layer.dilation[0] * (layer.kernel_size[0] - 1) + 1
        if isinstance(layer, torch.nn.Conv1d):
            # update size
            total_size += (layer_size - 1) * total_stride
            # update padding
            total_padding += layer.padding[0] * total_stride
            # update stride
            total_stride *= layer.stride[0]
        elif isinstance(layer, torch.nn.ConvTranspose1d):
            # update stride
            total_stride /= layer.stride[0]
            # update padding
            total_padding += (layer_size - layer.padding[0]) * total_stride
            # update size
            total_size += (layer_size - 1) * total_stride
    return total_size, total_stride, total_padding

@torch.inference_mode()
def compress(model, device, audio, win_duration, n_quantizers=None):
    """Encodes the given audio signal, returns the codes."""
    # right-pad to the next multiple of hop length
    # (as the model's internal padding is short by one hop length)
    remainder = audio.shape[-1] % model.hop_length
    right_pad = model.hop_length - remainder if remainder else 0
    if not win_duration:
        model.padding = True
        if right_pad:
            audio.zero_pad(0, right_pad)
        samples = audio.audio_data.to(device)
        codes = model.encode(samples, n_quantizers)["codes"]
        codes = codes.permute(2, 1, 0).short()  # -> time, quantizers, channels
    else:
        # determine receptive field of encoder
        model.padding = True
        field_size, stride, padding = receptive_field(model.encoder)
        model.padding = False
        # determine the window size to use
        # - the maximum samples the user wants to read at once
        win_size = int(win_duration * model.sample_rate)
        # - how many code frames we would get from this
        num_codes = (win_size - field_size + stride) // stride
        # - how many samples are actually involved in that
        win_size = field_size + (num_codes - 1) * stride
        # determine the hop size to use
        hop_size = num_codes * stride
        # finally process the input
        codes = []
        audio_size = audio.audio_data.size(-1)
        for start_position in tqdm.trange(-padding,
                                          audio_size + padding + right_pad,
                                          hop_size,
                                          leave=False):
            # extract chunk
            chunk = audio[..., max(0, start_position):start_position + win_size]
            # zero-pad the first chunk(s)
            if start_position < 0:
                chunk.zero_pad(-start_position, 0)
            chunk_size = chunk.audio_data.size(-1)
            # skip the last chunk if it would not have yielded any output
            if chunk_size + padding + right_pad < field_size:
                continue
            # pad the last chunk(s) to the full window size if needed
            if chunk_size < win_size:
                chunk.zero_pad(0, win_size - chunk_size)
            # process chunk
            samples = chunk.audio_data.to(device)
            c = model.encode(samples, n_quantizers)["codes"].cpu()
            c = c.permute(2, 1, 0)  # -> time, quantizers, channels
            # remove excess frames from padding if needed
            if chunk_size + padding + right_pad < win_size:
                chunk_codes = (chunk_size + padding + right_pad - field_size + stride) // stride
                c = c[:chunk_codes]
            codes.append(c.short())
        codes = torch.cat(codes, dim=0)
    return codes.contiguous()

def save_dac(outfile, codes, **metadata):
    """
    Writes the given codes to the given output file, with optional metadata.
    """
    try:
        with open(outfile, 'wb') as f:  # to allow a custom file extension
            np.savez(f, codes=codes, metadata=np.asarray(metadata, dtype='O'))
    except KeyboardInterrupt:
        Path(outfile).unlink()  # avoid half-written files
        raise

def load_dac(infile):
    """Reads codes and metadata from the given DAC file."""
    with np.load(infile, allow_pickle=True) as f:
        return torch.as_tensor(f['codes']), f['metadata'].item()

from dataclasses import dataclass
@dataclass
class DACArtifact:
    codes: torch.Tensor
    metadata: dict

    def save(self, outfile):
        save_dac(outfile, self.codes, **self.metadata)

    @classmethod
    def load(cls, infile):
        codes, metadata = load_dac(infile)
        return cls(codes=codes, metadata=metadata)

    def __repr__(self):
        return f"<DACArtifact {self.metadata}>"


class DACProcessor:

    def __init__(self, 
                 dac_path="./models/codec.pth", 
                 verbose: bool = False,
                 win_duration: float = 30.0):
        import torch
        from dac.utils import load_model as load_dac
        self.codec = load_dac(load_path=dac_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.codec.eval()
        self.codec.to(self.device)

        self.verbose = verbose
        self.win_duration = win_duration

    def process(self, sig: AudioSignal):
        sig = sig.to_mono()
        codes = compress(self.codec, self.device, sig, self.win_duration)
        # revert back to (channel, quantizer, time)
        codes = codes.permute(2, 1, 0).short()

        metadata = {
        }
        return DACArtifact(codes=codes, metadata=metadata)


def process_audio(conditioner, sig):
    outputs = conditioner.condition(sig)
    meta = outputs.pop("meta")
    features = ConditionFeatures(
        audio_path=str(sig.path_to_file),
        features={k: outputs[k].cpu().numpy() for k in outputs},
        metadata=meta
    ) 
    return features


def process_dac(conditioner, sig):
    artifact = conditioner.process(sig)
    return artifact


CONDITIONERS = [
    "dac", 
    "loudness"
]

def get_file_ext(name):
    return ".emb" if name != "dac" else ".adac"


def condition_and_save(
    input_csv: str = None,
    output_folder: str = None,
    num_workers: int = cpu_count(),
    overwrite: bool = False
):
    assert input_csv is not None, "input_csv must be specified"
    assert output_folder is not None, "output_folder must be specified"

    # Make a backup of the input csv
    shutil.copy(input_csv, f"{input_csv}.backup")
    print(f"Backed up {input_csv} to {input_csv}.backup")

    metadata = pd.read_csv(input_csv)
    # Scramble the metadata
    metadata = metadata.sample(frac=1)
    audio_files = [Path(p) for p in metadata['audio_path']]
    audio_roots = [Path(p) for p in metadata['audio_root']]
    assert all([r == audio_roots[0] for r in audio_roots]), "All audio roots must be the same"
    audio_root = Path(audio_roots[0])

    print(f"Found {len(audio_files)} audio files")

    conditioner_names = CONDITIONERS
    conditioners = {}
    for name in conditioner_names:
        if name == "dac":
            conditioners[name] = DACProcessor()
        else:
            conditioners[name] = REGISTRY[name]()

    for name in conditioner_names:
        file_ext = get_file_ext(name)
        if (f"{name}_path" not in metadata.columns) or overwrite:
            metadata[f"{name}_root"] = [output_folder for _ in range(len(metadata))]
            metadata[f"{name}_path"] = [None for _ in range(len(metadata))]
        else:
            if f"{name}_root" not in metadata.columns:
                metadata[f"{name}_root"] = [output_folder for _ in range(len(metadata))]
                print(f"WARNING: {name}_root not defined in {input_csv}.")
                print(f"Please define it in {input_csv} and re-run this script.")
                return

    dataset = AudioDataset(
        audio_files,
        audio_root,
        output_folder,
        conditioner_names,  # Pass conditioner names to the dataset
        file_ext=".tmp"  # Temporarily use a generic extension
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda x: x,
        prefetch_factor=4 if num_workers > 0 else None
    )

    def save(features, output_path):
        output_path.parent.mkdir(exist_ok=True, parents=True)
        features.save(output_path)

    with ThreadPoolExecutor(max_workers=(num_workers // 2) if num_workers > 0 else 1) as executor:
        print("Processing...")
        for batch in tqdm.tqdm(dataloader):
            sig = batch[0].to("cuda" if torch.cuda.is_available() else "cpu")
            idx = metadata[metadata['audio_path'] == str(sig.path_to_file)].index[0]

            if sig is None:
                print(f"skipping  since it could not be read")
                for name in conditioner_names:
                    idx = metadata[metadata['audio_path'] == str(sig.path_to_file)].index[0]
                    metadata.at[idx, f"{name}_path"] = None
                continue

            for name, conditioner in conditioners.items():
                file_ext = get_file_ext(name)
                if name == "dac":
                    output_path = Path(output_folder) / sig.path_to_file.with_suffix(file_ext)
                else:
                    output_path = Path(output_folder) / sig.path_to_file.parent / sig.path_to_file.stem / f"{name}.{file_ext}"

                if output_path.exists() and not overwrite:
                    print(f"skipping {output_path} since it already exists")
                    continue

                process_fn = process_dac if name == "dac" else process_audio
                features = process_fn(conditioner=conditioner, sig=sig)

                metadata.at[idx, f"{name}_path"] = str(output_path.relative_to(output_folder))

                executor.submit(save, features, output_path)

    print(f"Done! Writing to {input_csv}")
    metadata.to_csv(input_csv, index=False)
    print(f"All done! If you want to remove the backup, run `rm {input_csv}.backup`")

if __name__ == "__main__":
    DACProcessor = argbind.bind(DACProcessor)
    ChromaStemConditioner = argbind.bind(ChromaStemConditioner)
    YamnetConditioner = argbind.bind(YamnetConditioner)
    condition_and_save = argbind.bind(condition_and_save, without_prefix=True)
    args = argbind.parse_args()

    with argbind.scope(args):
        condition_and_save()

