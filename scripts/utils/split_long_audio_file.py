from pathlib import Path
import argbind

import audiotools as at
import tqdm


@argbind.bind(without_prefix=True)
def split_long_audio_file(
    file: str = None, 
    max_chunk_size_s: int = 60*10
):
    file = Path(file)
    output_dir = file.parent / file.stem
    output_dir.mkdir()
    
    sig = at.AudioSignal(file)

    # split into chunks
    for i, sig in tqdm.tqdm(enumerate(sig.windows(
        window_duration=max_chunk_size_s, hop_duration=max_chunk_size_s/2, 
        preprocess=True))
    ):
        sig.write(output_dir / f"{i}.wav")

    print(f"wrote {len(list(output_dir.glob('*.wav')))} files to {output_dir}")
    
    return output_dir

if __name__ == "__main__":
    args = argbind.parse_args()

    with argbind.scope(args):
        split_long_audio_file()