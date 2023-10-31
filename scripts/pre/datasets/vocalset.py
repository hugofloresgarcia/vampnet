""""create a basic metadata.csv from a folder of audio files"""

from pathlib import Path
import pandas as pd
import audiotools as at

import argbind
import tqdm

def match(p1, p2):
    return (str(p1.stem)[0:3]) == (str(p2.stem)[0:3])

@argbind.bind(without_prefix=True)
def create_vocalset_dataset(
    vocalset_folder: str = None,
    output_file: str = None,
):
    """Main function to generate metadata CSV for audio files.

    Parameters:
    - vocalset_folder (str): Path to the directory containing the audio files.
    - output_file (str): Path where the metadata CSV will be saved.

    Returns:
    None: Saves the metadata CSV to the specified output path.
    """
    assert vocalset_folder is not None, "Must provide vocalset_folder"
    assert output_file is not None, "Must provide output_file"
    vocalset_folder = Path(vocalset_folder)

    print(f"looking for audio files in vocal folder")
    vocal_files = at.util.find_audio(
        vocalset_folder / "vocal_imitations" / "included"  , ext=[".wav", ".mp3", ".flac", ".WAV", ".MP3", ".FLAC"]
    )
    print(f"Found {len(vocal_files)} vocal files")

    print(f"looking for audio files in sources folder")
    sources_files = at.util.find_audio(
        vocalset_folder / "original_recordings" / "reference", ext=[".wav", ".mp3", ".flac", ".WAV", ".MP3", ".FLAC"]
    )
    print(f"Found {len(sources_files)} source files")

    def id_(path):
        return str(path)[0:3]

    # go through each source file and find all the matching vocal files
    pairs = []
    for src_file in tqdm.tqdm(sources_files):
        for vocal_file in vocal_files:
            if match(src_file, vocal_file):
                pairs.append((vocal_file, src_file))

    print(f"found {len(pairs)} pairs")
    
    # ceate two dataframes, one for the vocals and one for the sources
    # match them with an ID
    vocal_metadata = []
    sources_metadata = []
    for i, (vocal_file, src_file) in enumerate(pairs):
        vocal_metadata.append({
            "audio_path": str(Path(vocal_file).absolute().relative_to(Path(vocalset_folder))),
            "audio_root": str(Path(vocalset_folder).absolute()),
            "id": i,
        })
        sources_metadata.append({
            "audio_path": str(Path(src_file).absolute().relative_to(Path(vocalset_folder))),
            "audio_root": str(Path(vocalset_folder).absolute()),
            "id": i,
        })
    
    vocal_metadata = pd.DataFrame(vocal_metadata)
    sources_metadata = pd.DataFrame(sources_metadata)

    # merge the dataframes and add a "type" column that says whether it's input or output
    metadata = pd.concat([vocal_metadata, sources_metadata])
    metadata['type'] = metadata['audio_path'].apply(lambda x: "vocal" if "vocal_imitations" in str(x) else "source")

    # save the metadata
    metadata.to_csv(output_file, index=False)

if __name__ == "__main__":
    args = argbind.parse_args()

    with argbind.scope(args):
        create_vocalset_dataset()