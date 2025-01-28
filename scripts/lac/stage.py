import os
import subprocess
from pathlib import Path

import argbind
import rich
from audiotools.ml import Experiment


@argbind.bind(without_prefix=True)
def run(
    run_dir: str = None,
    name: str = None,
    shell: str = "zsh",
    no_snapshot: bool = False,
    recent: bool = False,
):
    if run_dir is None:
        run_dir = os.getenv("PATH_TO_RUNS", "runs")
    if recent:
        paths = sorted(Path(run_dir).iterdir(), key=os.path.getmtime)
        paths = [p.name for p in paths if p.is_dir()]
        if paths:
            name = paths[-1]

    with Experiment(run_dir, name) as exp:
        if not no_snapshot:
            exp.snapshot()
            rich.print(f"Created a snapshot of {exp.parent_directory} at {exp.exp_dir}")

        rich.print(f"Switched working directory to {exp.exp_dir}")
        rich.print("Spawning shell")
        subprocess.call(shell)


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        run()
