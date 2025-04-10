from pathlib import Path

run_dir = Path("runs/lazaro-ros-sep")
name = run_dir.name

repo_dir = Path("models/vampnet")

for part in ("coarse", "c2f"):
    outdir = repo_dir / "loras" / name 
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"{part}.pth"
    path = run_dir / part / "latest" / "vampnet" / "weights.pth"
    path.rename(outpath)
    print(f"moved {path} to {outpath}")

# now, push to hub
from huggingface_hub import Repository
repo = Repository(str(repo_dir),  git_user="hugofloresgarcia", git_email="huferflo@gmail.com")
repo.push_to_hub(
    commit_message=f"add {name}"
)