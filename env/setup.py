# This script guides the user through setting up their env.sh
# if env.sh does not exist. Should have no dependencies other
# than Python standard library.
import shlex
import socket
import subprocess
import textwrap


def run(cmd):
    return subprocess.check_output(shlex.split(cmd)).decode("utf-8")


print("1. Setting up Google Cloud access")
print("---------------------------------")
gcloud_authorized = "gs://research-data-raw" in run("gsutil ls")
if not gcloud_authorized:
    run("gcloud auth login")

run("gcloud config set project lyrebird-research")
run("gcloud auth configure-docker")

print()
print("2. Setting up Github access")
print("---------------------------")

lines = textwrap.wrap(
    "First, let's get your Github token, so all "
    "packages can be installed. Create one by going to your "
    "Github profile -> Developer settings -> Personal access tokens -> "
    "Generate new token. Copy the token below."
)
[print(l) for l in lines]

GITHUB_TOKEN = input("\nGithub token: ") or "undefined"

print()
print("3. Setting up Jupyter and Tensorboard")
print("-------------------------------------")

JUPYTER_TOKEN = input("Password for Jupyter server (default:password): ") or "password"
JUPYTER_PORT = input("Jupyter port to run on (default:8888): ") or "8888"
TENSORBOARD_PORT = input("Tensorboard port to run on (default:6006): ") or "6006"

print()
print("4. Setting up paths.")
print("--------------------")

PATH_TO_RUNS = input("Where runs should go (default:./runs/): ") or "./runs/"
TENSORBOARD_PATH = (
    input("Bucket/dir for tensorboard logs (default=PATH_TO_RUNS): ") or PATH_TO_RUNS
)

with open("env/data.sh") as f:
    data_script = f.read()

write_to_data_sh = False
if socket.gethostname() not in data_script:
    print("Looks like the data path for this machine is not setup.")
    PATH_TO_DATA = input(f"Path to data on {socket.gethostname()}: ") or "~/data"

    data_command = f"""
if [[ $(hostname) == "{socket.gethostname()}" ]]; then
    export PATH_TO_DATA={PATH_TO_DATA}
fi
    """
    write_to_data_sh = True


print()
print("5. Setting up Papaya")
print("-----------------------------------------")

PAPAYA_USER_TOKEN = input("Papaya user token: ") or "undefined"

env_script = f"""
source env/alias.sh
source env/data.sh
export GITHUB_TOKEN={GITHUB_TOKEN}

export PAPAYA_USER_TOKEN={PAPAYA_USER_TOKEN}

export HOST_USER_ID=$(id -u)
export HOST_USER_GID=$(id -g)

export JUPYTER_TOKEN={JUPYTER_TOKEN}
export JUPYTER_PORT={JUPYTER_PORT}
export TENSORBOARD_PORT={TENSORBOARD_PORT}

export PATH_TO_RUNS={PATH_TO_RUNS}
export TENSORBOARD_PATH={TENSORBOARD_PATH}
"""

print()
print("6. Potential file contents.")
print("---------------------------")

print("env/env.sh: \n")
print("##################")
print(env_script)
print("##################")

if write_to_data_sh:
    data_script += data_command

print("env/data.sh:")
print("##################")
print(data_script)
print("##################")

print()
write_to_files = input("Write to file [yn]? ") or "n"
if write_to_files == "y":
    with open("env/env.sh", "w") as f:
        f.write(env_script.strip())
    with open("env/data.sh", "w") as f:
        f.write(data_script.strip())

print()
print("8. Finalize setup.")
print("------------------")
print("Run the following command to complete setup.")
print("source env/env.sh")
