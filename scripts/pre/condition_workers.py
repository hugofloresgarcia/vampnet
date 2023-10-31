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
                 dac_path="./models/dac/weights.pth", 
                 verbose: bool = False,
                 batch_size: int = 10, 
                 win_duration: float = 10.0):
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
    features = ConditionFeatures(
        audio_path=str(sig.path_to_file),
        features={k: outputs[k].cpu().numpy() for k in outputs},
        metadata={}
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
    num_workers: int = cpu_count()
):
    assert input_csv is not None, "input_csv must be specified"
    assert output_folder is not None, "output_folder must be specified"

    # make a backup of the input csv
    shutil.copy(input_csv, f"{input_csv}.backup")
    print(f"backed up {input_csv} to {input_csv}.backup")

    metadata = pd.read_csv(input_csv)
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
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=lambda x: x,prefetch_factor=4)

    output_files = []
    for batch in tqdm(dataloader):
        sig = batch[0]
        if sig is None:
            print(f"skipping  since it could not be read")
            output_files.append(None)
            continue

        audio_file = sig.path_to_file
        output_path = Path(output_folder) / audio_file.with_suffix(file_ext)

        # This is not strictly necessary, since we've already filtered
        # out existing files in the AudioDataset, but it can be left in for clarity.
        if output_path.exists():
            print(f"skipping {audio_file.name} since {output_path} already exists")
            output_files.append(str(output_path.relative_to(output_folder)))
            continue

        process_fn = process_dac if conditioner_name == "dac" else process_audio
        features = process_fn(conditioner=conditioner, sig=sig)

        output_path.parent.mkdir(exist_ok=True, parents=True)
        features.save(output_path)
        
        output_files.append(str(output_path.relative_to(output_folder)))

    # write to the output csv
    metadata[f"{conditioner_name}_path"] = output_files
    metadata[f"{conditioner_name}_root"] = [output_folder for _ in range(len(metadata))]

    print(f"done! writing to {input_csv}")
    metadata.to_csv(input_csv, index=False)

    print(f"all done! if you want to remove the backup, run `rm {input_csv}.backup`")

if __name__ == "__main__":
    args = argbind.parse_args()

    with argbind.scope(args):
        condition_and_save()

