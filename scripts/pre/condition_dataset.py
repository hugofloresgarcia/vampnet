from pathlib import Path
import tqdm
import argbind
import audiotools as at
from audiotools import AudioSignal

from vampnet.condition import REGISTRY, ConditionFeatures, ChromaStemConditioner, YamnetConditioner

@argbind.bind(without_prefix=True)
def condition_and_save(
    audio_folder: str = None, 
    output_folder: str= None,
    conditioner_name: str = "yamnet",
):
    assert audio_folder is not None, "audio_folder must be specified"
    assert output_folder is not None, "output_folder must be specified"

    audio_files = list(at.util.find_audio(Path(audio_folder)))
    print(f"Found {len(audio_files)} audio files ")

    conditioner = REGISTRY[conditioner_name]()

    for audio_file in tqdm.tqdm(audio_files, desc=f"embedding conditioner {conditioner_name}"):
        # get the output filepath relative to the audio folder
        output_path = audio_file.relative_to(audio_folder)

        if output_path.exists():
            print(f"skipping {audio_file.name} because it already exists")
            continue
        else: 
            try:
                sig = AudioSignal(audio_file)
            except Exception as e:
                print(f"failed to load {audio_file.name} with error {e}")
                print(f"skipping {audio_file.name}")
                continue


            features = ConditionFeatures(
                audio_path=str(audio_file.relative_to(audio_folder)), 
                features=conditioner.condition(sig).cpu().numpy(),
                metadata={}
            )
            
            # cache the embeddings
            output_path.parent.mkdir(exist_ok=True, parents=True)
            features.save(output_path)

    
if __name__ == "__main__":
    args = argbind.parse_args()

    with argbind.scope(args):
        condition_and_save()



