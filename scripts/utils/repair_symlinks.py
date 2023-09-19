import pathlib

def recreate_symlinks(data_root_dir, local_root_dir):
    data_root = pathlib.Path(data_root_dir)
    local_root = pathlib.Path(local_root_dir)

    # Check if provided paths are directories
    if not data_root.is_dir() or not local_root.is_dir():
        print("Both paths should point to directories!")
        return

    # Iterate through all the files in the data root directory
    for data_path in data_root.rglob('*'):
        if data_path.is_file():
            # Corresponding path in the local root directory
            relative_path = data_path.relative_to(data_root)
            local_path = local_root / relative_path

            # If the local path is not a symlink or it's broken, replace it with a new symlink
            if not local_path.exists():
                print(f"doesn't exist. ")
            if not local_path.is_symlink():
                # First, remove the current file if it exists
                if local_path.exists():
                    print(f"Removing broken or non-symlink file: {local_path}")
                    local_path.unlink()
mv 
                # Make sure parent directory exists
                local_path.parent.mkdir(parents=True, exist_ok=True)

                # Create symlink
                print(f"Creating symlink: {local_path} -> {data_path}")
                local_path.symlink_to(data_path)

    print("Symlinks recreation process completed!")

if __name__ == "__main__":
    data_root_dir = "/media/CHONK/hugo/fma/fma_full"
    local_root_dir = input("Enter the local root directory path: ").strip()

    recreate_symlinks(data_root_dir, local_root_dir)