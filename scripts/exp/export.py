from pathlib import Path

import shutil
import argparse
from vampnet import DEFAULT_HF_MODEL_REPO
from huggingface_hub import create_repo, repo_exists, HfApi



parser = argparse.ArgumentParser(description="Export the fine-tuned model to the repo")
parser.add_argument(
    "--name", type=str, default="lazaro-ros-sep",
    help="name of the fine-tuned model to export"
)
parser.add_argument(
    "--model", type=str, default="latest",
    help="model version to export. check runs/<name> for available versions"
)
parser.add_argument(
    "--repo", type=str, default=DEFAULT_HF_MODEL_REPO,
    help="name of the repo to export to"
)

args = parser.parse_args()
name = args.name
version = args.model

##
print(f"~~~~~~~~~~~ vampnet export! ~~~~~~~~~~~~")
print(f"exporting {name} version {version} to {args.repo}\n")

run_dir = Path(f"runs/{name}")
repo_dir = Path("models/vampnet")

# create our repo
new_repo = False
if not repo_exists(args.repo):
    print(f"repo {args.repo} does not exist, creating it")
    print(f"creating a repo at {args.repo}")
    create_repo(args.repo)
    new_repo = True

paths = []
for part in ("coarse", "c2f"):
    outdir = repo_dir / "loras" / name 
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"{part}.pth"
    path = run_dir / part / version / "vampnet" / "weights.pth"
    # path.rename(outpath)
    shutil.copy(path, outpath)
    paths.append(outpath)
    print(f"copied {path} to {outpath}")

print(f"uploading files to {args.repo}")
# upload files to the repo

# if it's a new repo, let's add the default models too
if new_repo:
    paths.extend([repo_dir / "c2f.pth", repo_dir / "coarse.pth", repo_dir / "codec.pth", repo_dir / "wavebeat.pth"])

api = HfApi()

for path in paths:
    path_in_repo = str(path.relative_to(repo_dir))
    print(f"uploading {path} to {args.repo}/{path_in_repo}")
    api.upload_file(
        path_or_fileobj=path,
        path_in_repo=path_in_repo,
        repo_id=args.repo,
        token=True,
        commit_message=f"uploading {path_in_repo}",
    )


print("done!!! >::0")