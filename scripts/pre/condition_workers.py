import random
from pathlib import Path
import argbind
import audiotools as at
from audiotools import AudioSignal
from multiprocessing import cpu_count 
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from tqdm import tqdm
import shutil
import pandas as pd

from vampnet.condition import REGISTRY, ConditionFeatures, ChromaStemConditioner, YamnetConditioner
from torch.utils.data import Dataset, DataLoader

class AudioDataset(Dataset):
    def __init__(self, audio_files, audio_root, output_folder, conditioner_name, file_ext=".emb"):
        self.audio_root = audio_root
        self.output_folder = output_folder
        self.conditioner_name = conditioner_name
        self.file_ext = file_ext if conditioner_name != "dac" else ".dac"
        
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


class DACProcessor:

    def __init__(self, 
                 dac_path="./models/codec.pth", 
                 verbose: bool = False,
                 batch_size: int = 15, 
                 win_duration: float = 20.0):
        import torch
        from dac.utils import load_model as load_dac
        self.codec = load_dac(load_path=dac_path)
        self.codec.eval()
        self.codec.to("cuda" if torch.cuda.is_available() else "cpu")

        self.verbose = verbose
        self.batch_size = batch_size
        self.win_duration = win_duration

    def process(self, sig: AudioSignal):
        sig = sig.to_mono()
        artifact = self.codec.compress(
            sig, 
            verbose=self.verbose,
            win_duration=self.win_duration,
            win_batch_size=self.batch_size
        )
        artifact.codes.cpu()
        return artifact


DACProcessor = argbind.bind(DACProcessor)
ChromaStemConditioner = argbind.bind(ChromaStemConditioner)
YamnetConditioner = argbind.bind(YamnetConditioner)


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


@argbind.bind(without_prefix=True)
def condition_and_save(
    input_csv: str = None,
    output_folder: str= None,
    conditioner_name: str = "dac",
    num_workers: int = cpu_count(), 
    overwrite: bool = False
):
    assert input_csv is not None, "input_csv must be specified"
    assert output_folder is not None, "output_folder must be specified"

    # make a backup of the input csv
    shutil.copy(input_csv, f"{input_csv}.backup")
    print(f"backed up {input_csv} to {input_csv}.backup")

    metadata = pd.read_csv(input_csv)
    # scramble the rows
    metadata = metadata.sample(frac=1).reset_index(drop=True)
    audio_files = [Path(p) for p in metadata['audio_path']]
    audio_roots = [Path(p) for p in metadata['audio_root']]
    assert all([r == audio_roots[0] for r in audio_roots]), "all audio roots must be the same"
    audio_root = Path(audio_roots[0])

    print(f"Found {len(audio_files)} audio files")

    if conditioner_name == "dac":
        conditioner = DACProcessor()
    else:
        conditioner = REGISTRY[conditioner_name]()

    file_ext = ".emb" if conditioner_name != "dac" else ".dac"

    dataset = AudioDataset(audio_files, audio_root, output_folder, conditioner_name)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=lambda x: x,prefetch_factor=8 if num_workers > 0 else None)

    if (f"{conditioner_name}_path" not in metadata.columns) or overwrite:
        metadata[f"{conditioner_name}_root"] = [output_folder for _ in range(len(metadata))]
        metadata[f"{conditioner_name}_path"] = [None for _ in range(len(metadata))]
    else:
        # make sure the conditioner root is defined too. otherwise, prompt the user to define it
        if f"{conditioner_name}_root" not in metadata.columns:
            metadata[f"{conditioner_name}_root"] = [output_folder for _ in range(len(metadata))]
            print(f"WARNING: {conditioner_name}_root not defined in {input_csv}.")
            print(f"Please define it in {input_csv} and re-run this script.")
            return

    # go through the audio files, get the output_path, place it on the metadata (if it exists)
    print(f"checking for existing files...")
    for idx, row in tqdm(metadata.iterrows()):
        audio_file = Path(row['audio_path'])
        output_path = Path(output_folder) / audio_file.with_suffix(file_ext)
        if output_path.exists():
            metadata.at[idx, f"{conditioner_name}_path"] = str(output_path.relative_to(output_folder))

    def save(features, output_path):
        output_path.parent.mkdir(exist_ok=True, parents=True)
        features.save(output_path)
        
    with ThreadPoolExecutor(max_workers=(num_workers // 2) if num_workers > 0 else 1) as executor:
        # now, process the non-existent files
        print("processing...")
        for batch in tqdm(dataloader):
            sig = batch[0]
            if sig is None:
                print(f"skipping  since it could not be read")
                metadata.at[idx, f"{conditioner_name}_path"] = None
                continue

            audio_file = sig.path_to_file
            output_path = Path(output_folder) / audio_file.with_suffix(file_ext)

            process_fn = process_dac if conditioner_name == "dac" else process_audio
            features = process_fn(conditioner=conditioner, sig=sig)

            idx = metadata[metadata['audio_path'] == str(sig.path_to_file)].index[0]
            metadata.at[idx, f"{conditioner_name}_path"] = str(output_path.relative_to(output_folder))

            executor.submit(save, features, output_path)
        
        print(f"done! writing to {input_csv}")
        metadata.to_csv(input_csv, index=False)

    print(f"all done! if you want to remove the backup, run `rm {input_csv}.backup`")

if __name__ == "__main__":
    args = argbind.parse_args()

    with argbind.scope(args):
        condition_and_save()