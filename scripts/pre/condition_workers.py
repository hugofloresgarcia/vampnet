from pathlib import Path
import argbind
import audiotools as at
from audiotools import AudioSignal
from multiprocessing import cpu_count 
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from tqdm import tqdm

from vampnet.condition import REGISTRY, ConditionFeatures, ChromaStemConditioner, YamnetConditioner

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

def load_audio(kwargs):
    audio_file = kwargs['audio_file']
    output_path = kwargs['output_path']
    
    if output_path.exists():
        # print(f"Skipping {audio_file.name} (already exists).")
        return audio_file, None
    
    try:
        sig = AudioSignal(audio_file)
    except Exception as e:
        print(f"failed to load {audio_file.name} with error {e}")
        print(f"skipping {audio_file.name}")
        return audio_file, None

    return audio_file, sig


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


# TODO: update to work with csv metadata
@argbind.bind(without_prefix=True)
def condition_and_save(
    audio_folder: str = None,
    output_folder: str= None,
    conditioner_name: str = "yamnet",
    processes: int = cpu_count()
):
    assert audio_folder is not None, "audio_folder must be specified"
    assert output_folder is not None, "output_folder must be specified"

    audio_files = list(at.util.find_audio(Path(audio_folder)))
    import random
    random.shuffle(audio_files)

    if conditioner_name == "dac":
        conditioner = DACProcessor()
    else:
        conditioner = REGISTRY[conditioner_name]()

    file_ext = ".emb" if conditioner_name != "dac" else ".dac"
    load_audio_args = [{
        'audio_file': audio_file, 
        'output_path': Path(output_folder) / audio_file.relative_to(audio_folder).with_suffix(file_ext)
    } for audio_file in audio_files]

    if processes == 0:
        for kwargs in tqdm(load_audio_args):
            audio_file, sig = load_audio(kwargs)
            output_path = kwargs['output_path']

            if sig is None:
                # print(f"skipping {audio_file.name}")
                continue
            sig.path_to_file = audio_file.relative_to(audio_folder)
            process_fn = process_dac if conditioner_name == "dac" else process_audio
            features = process_fn(
                conditioner=conditioner, 
                sig=sig)
            

            output_path.parent.mkdir(exist_ok=True, parents=True)
            features.save(output_path)
    else:
        with ThreadPoolExecutor(max_workers=processes//2) as save_executor:
            with ProcessPoolExecutor(max_workers=processes) as executor:
                print(f"submitting futures")
                # print(f"submitted {len(futures)} futures")

                pbar = tqdm(as_completed(executor.submit(load_audio, kwargs) for kwargs in load_audio_args), total=len(load_audio_args))
                for future in pbar:
                    try:
                        # print("got future")
                        audio_file, sig = future.result()
                        if sig is None:
                            print(f"skipping {audio_file.name}")
                            continue
                        pbar.set_description(f"Processing {audio_file.name}")
                        sig.path_to_file = audio_file.relative_to(audio_folder)

                        # are we processing dac? if so, use a different process fn
                        process_fn = process_dac if conditioner_name == "dac" else process_audio
                        features = process_fn(
                            conditioner=conditioner, 
                            sig=sig)
                        
                        pbar.set_description(f"Saving {audio_file.name}")
                        def save():
                            file_ext = ".emb" if conditioner_name != "dac" else ".dac"
                            output_path = Path(output_folder) / audio_file.relative_to(audio_folder).with_suffix(file_ext)
                            output_path.parent.mkdir(exist_ok=True, parents=True)
                            features.save(output_path)

                        save_executor.submit(save)
                        del future

                    except Exception as e:
                        print(f"failed to process {audio_file.name} with error {e}")
                        print(f"skipping {audio_file.name}")
                        continue




if __name__ == "__main__":
    args = argbind.parse_args()

    with argbind.scope(args):
        condition_and_save()

