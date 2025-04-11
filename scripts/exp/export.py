from pathlib import Path

import shutil
import argparse
from vampnet import DEFAULT_HF_MODEL_REPO


parser = argparse.ArgumentParser(description="Export the fine-tuned model to the repo")
parser.add_argument(
    "--name", type=str, default="lazaro-ros-sep",
    help="name of the fine-tuned model to export"
)
parser.add_argument(
    "--model", type=str, default="latest",
    help="model version to export. check runs/<name> for available versions"
)

args = parser.parse_args()
name = args.name
version = args.model

run_dir = Path(f"runs/{name}")
repo_dir = Path("models/vampnet")

for part in ("coarse", "c2f"):
    outdir = repo_dir / "loras" / name 
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"{part}.pth"
    path = run_dir / part / version / "vampnet" / "weights.pth"
    # path.rename(outpath)
    shutil.copy(path, outpath)
    print(f"moved {path} to {outpath}")

from huggingface_hub import Repository
repo = Repository(str(repo_dir),clone_from=f"https://huggingface.co/{DEFAULT_HF_MODEL_REPO}")
print(f"pushing {repo_dir} to {name}")
print(f"howdy! I am pushing to {DEFAULT_HF_MODEL_REPO} :)")
repo.push_to_hub(
    commit_message=f"add {name}", 
)
print("done!!! >::0")