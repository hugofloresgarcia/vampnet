#!/usr/bin/env python3
"""
VampNet Launcher: orchestrates a single SSH session (with port-forwarding
and real-time remote stdout/stderr), local client startup, and Max patch loading.
Supports a --hold mode to delay cleanup until this script is killed, even after Max exits.
"""
import argparse
import json
import logging
import signal
import subprocess
import sys
import threading
from pathlib import Path


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)


class VampNetLauncher:
    def __init__(self, config_path: Path, hold: bool = False):
        self.logger = setup_logger()
        self.config = self._load_config(config_path)
        self.ssh_proc = None
        self.client_proc = None
        self.max_proc = None
        self.hold = hold

        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda s, f: self._handle_exit())

    @staticmethod
    def _load_config(path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        cfg = json.loads(path.read_text())
        required = ('host', 'remote_python', 'remote_dir', 'port', 'maxpat')
        missing = [k for k in required if k not in cfg]
        if missing:
            raise KeyError(f"Missing config keys: {', '.join(missing)}")
        if cfg['host'] == "user@your-server" or cfg['remote_python'] == "/path/to/python" or cfg['remote_dir'] == "/path/to/vampnet":
            raise ValueError("First update host, remote_python and _remote_dir in config.json.")
        return cfg

    @staticmethod
    def _stream_pipe(pipe, prefix, log_fn):
        try:
            for line in iter(pipe.readline, b""):
                log_fn(f"{prefix} {line.decode().rstrip()}")
        finally:
            pipe.close()

    def _run_remote(self):
        port = int(self.config['port'])
        remote_dir = self.config['remote_dir']
        remote_py  = self.config['remote_python']
        cmd = [
            "ssh",
            "-tt", # Force pseudo-terminal allocation, so that app.py is killed once the ssh session ends
            "-o", "ExitOnForwardFailure=yes",
            "-L", f"{port}:localhost:{port}",
            self.config['host'],
            f"bash -lc \"cd {remote_dir} && exec {remote_py} -u app.py "
            f"--args.load conf/interface.yml --Interface.device cuda\""
        ]
        self.logger.info(f"SSH+remote cmd: {' '.join(cmd)}")
        self.ssh_proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        # Demote stderr to INFO
        threading.Thread(
            target=self._stream_pipe,
            args=(self.ssh_proc.stderr, "[Remote][ERR]", self.logger.info),
            daemon=True
        ).start()

        # Read stdout until readiness marker appears
        ready_marker = "Running on local URL"
        for raw in self.ssh_proc.stdout:
            line = raw.decode().rstrip()
            self.logger.info(f"[Remote] {line}")
            if ready_marker in line:
                self.logger.info("✔ Remote service is ready")
                break

        # Drain remaining stdout asynchronously
        threading.Thread(
            target=self._stream_pipe,
            args=(self.ssh_proc.stdout, "[Remote]", self.logger.info),
            daemon=True
        ).start()

    def _teardown(self):
        self.logger.info("Tearing down processes...")
        # kill SSH + remote app
        if self.ssh_proc and self.ssh_proc.poll() is None:
            self.logger.info(f"Terminating SSH/remote (PID {self.ssh_proc.pid})")
            self.ssh_proc.terminate()
            try:
                self.ssh_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.logger.warning("SSH/remote did not exit; killing")
                self.ssh_proc.kill()

        # kill local client
        if self.client_proc and self.client_proc.poll() is None:
            self.logger.info(f"Terminating client (PID {self.client_proc.pid})")
            self.client_proc.terminate()
            try:
                self.client_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.logger.warning("Client did not exit; killing")
                self.client_proc.kill()

        # kill Max
        if self.max_proc and self.max_proc.poll() is None:
            self.logger.info(f"Terminating Max (PID {self.max_proc.pid})")
            self.max_proc.terminate()
            try:
                self.max_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.logger.warning("Max did not exit; killing")
                self.max_proc.kill()

    def _handle_exit(self):
        self._teardown()
        sys.exit(0)

    def run(self):
        root   = Path(__file__).resolve().parent.parent
        patch  = root / self.config['maxpat']
        client = root / 'unloop' / 'client.py'

        # Preflight checks
        missing = []
        if not patch.exists():   missing.append(f"Max patch not found: {patch}")
        if not client.exists():  missing.append(f"Client script not found: {client}")
        if missing:
            for m in missing:
                self.logger.error(m)
            raise FileNotFoundError("Required files missing, aborting launch.")

        try:
            # 1) SSH + remote app + port-forward + log streaming
            self._run_remote()

            # 2) Launch local client
            port = int(self.config['port'])
            client_cmd = [
                sys.executable, str(client),
                '--vampnet_url', f"http://127.0.0.1:{port}/"
            ]
            self.logger.info(f"Client cmd: {' '.join(client_cmd)}")
            self.client_proc = subprocess.Popen(
                client_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            threading.Thread(
                target=self._stream_pipe,
                args=(self.client_proc.stdout, "[Client]", self.logger.info),
                daemon=True
            ).start()
            threading.Thread(
                target=self._stream_pipe,
                args=(self.client_proc.stderr, "[Client][ERR]", self.logger.info),
                daemon=True
            ).start()

            # 3) Open Max and wait for it to exit
            self.logger.info(f"Opening Max patch and waiting: {patch}")
            self.max_proc = subprocess.Popen(
                ["open", "-n", "-W", "-a", "Max", str(patch)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            threading.Thread(
                target=self._stream_pipe,
                args=(self.max_proc.stdout, "[Max]", self.logger.info),
                daemon=True
            ).start()
            threading.Thread(
                target=self._stream_pipe,
                args=(self.max_proc.stderr, "[Max][ERR]", self.logger.info),
                daemon=True
            ).start()

            self.max_proc.wait()

            # 4) After Max exits, keep the launcher alive until it's killed
            if self.hold:
                # prevent SIGCHLD (child exit) from waking us up
                signal.signal(signal.SIGCHLD, signal.SIG_IGN)
                self.logger.info("Max exited; awaiting SIGINT/SIGTERM to clean up.")
                while True:
                    signal.pause()

        except (subprocess.SubprocessError, RuntimeError, ValueError) as e:
            self.logger.error(f"Launcher error: {e}")
        finally:
            self._teardown()


def main():
    parser = argparse.ArgumentParser(description="Launch VampNet with JSON config.")
    parser.add_argument(
        '--config',    required=True,      help='Path to JSON config'
    )
    parser.add_argument(
        '--hold', action='store_true',
        help='Delay cleanup until this script is killed, even after Max exits.'
    )
    args = parser.parse_args()
    VampNetLauncher(Path(args.config), hold=args.hold).run()


if __name__ == '__main__':
    main()
