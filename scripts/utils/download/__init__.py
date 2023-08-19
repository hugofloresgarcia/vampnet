import argbind
from pathlib import Path
from typing import Union
import subprocess


def download_file(url, filename: Union[bool, str] = False, verbose = False):
    # __author__  = "github.com/ruxi"
    # __license__ = "MIT"
    """
    Download file with progressbar
    
    Usage:
        download_file('http://web4host.net/5MB.zip')  
    """
    import requests 
    import tqdm     # progress bar
    import os.path
    if not filename:
        local_filename = os.path.join(".",url.split('/')[-1])
    else:
        local_filename = filename
    r = requests.get(url, stream=True)
    file_size = int(r.headers['Content-Length'])
    chunk = 1
    chunk_size=1024
    num_bars = int(file_size / chunk_size)
    if verbose:
        print(dict(file_size=file_size))
        print(dict(num_bars=num_bars))

    with open(local_filename, 'wb') as fp:
        for chunk in tqdm.tqdm(
                                    r.iter_content(chunk_size=chunk_size)
                                    , total= num_bars
                                    , unit = 'KB'
                                    , desc = local_filename
                                    , leave = True # progressbar stays
                                ):
            fp.write(chunk)
    return



def download(url, dest_dir):
    # check if the file already exists, prompt to overwrite
    print(f"downloading {url} to {dest_dir}")
    filename = url.split("/")[-1]
    dest_file = dest_dir / filename
    if dest_file.exists():
        print(f"{dest_file} already exists, overwrite? [y/N]")
        if input() == "y":
            download_file(url, filename=str(dest_file))
            print(f"succesfully downloaded {dest_file}!")
        else:
            print(f"skipping download of {dest_file}")
    else:
        download_file(url, filename=str(dest_file))
        print(f"succesfully downloaded {dest_file}!")

    return dest_file

def unzip(dest_file, dest_dir):
    # unzip the file
    print(f"unzipping {dest_file}")
    subprocess.run(["7z", "x", str(dest_file), f"-o{str(dest_dir)}"])
