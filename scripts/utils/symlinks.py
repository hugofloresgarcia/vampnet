import pathlib

def create_audio_symlinks(codes_dir, audio_dir, symlink_dir):
    codes_path = pathlib.Path(codes_dir)
    audio_path = pathlib.Path(audio_dir)
    symlink_path = pathlib.Path(symlink_dir)

    # Check if provided paths for codes and audio are directories
    if not codes_path.is_dir() or not audio_path.is_dir():
        print("Both codes and audio paths should point to directories!")
        return

    # Create symlink_dir if it doesn't exist
    symlink_path.mkdir(parents=True, exist_ok=True)

    # Iterate through all .dac files in the codes directory
    for dac_file in codes_path.rglob('*.dac'):
        # Get the relative path of the dac file to mimic the structure in audio_dir
        relative_path = dac_file.relative_to(codes_path)

        # Replace .dac extension with .wav and .mp3 to look for the audio file
        wav_file = audio_path / relative_path.with_suffix('.wav')
        mp3_file = audio_path / relative_path.with_suffix('.mp3')

        if wav_file.exists():
            audio_file = wav_file
        elif mp3_file.exists():
            audio_file = mp3_file
        else:
            print(f"No corresponding audio file found for: {dac_file}")
            continue

        # Create the corresponding symlink path
        symlink_file_path = symlink_path / relative_path.with_suffix(audio_file.suffix)

        # Make sure parent directory exists
        symlink_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Create symlink
        print(f"Creating symlink: {symlink_file_path} -> {audio_file}")
        symlink_file_path.symlink_to(audio_file)

    print("Symlink creation process completed!")

if __name__ == "__main__":
    codes_dir = "data/codes/prosound/train"
    audio_dir = "/media/CHONK2/prosound_core_complete"
    symlink_dir = "data/prosound/train"

    create_audio_symlinks(codes_dir, audio_dir, symlink_dir)