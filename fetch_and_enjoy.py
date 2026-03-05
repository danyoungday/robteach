"""SCP a checkpoint from a remote Linux machine and visualize locally.

Runs on the Mac. Fetches both the model .zip and VecNormalize .pkl,
then execs enjoy.py via mjpython.

Usage:
    mjpython fetch_and_enjoy.py danyoungday@1.2.3.4
    mjpython fetch_and_enjoy.py danyoungday@1.2.3.4 logs/checkpoints/ppo_50000_steps
    mjpython fetch_and_enjoy.py danyoungday@1.2.3.4 --episodes 5
    mjpython fetch_and_enjoy.py danyoungday@1.2.3.4 --remote-dir ~/other/project
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def _remote_file_exists(ssh_host: str, remote_path: str) -> bool:
    """Check if a file exists on the remote machine via ssh."""
    result = subprocess.run(
        ["ssh", ssh_host, "test", "-f", remote_path],
        capture_output=True,
    )
    return result.returncode == 0


def _find_remote_vec_normalize(ssh_host: str, remote_dir: str, checkpoint: str) -> str:
    """Mirror the _find_vec_normalize logic from enjoy.py, but over SSH."""
    ckpt = Path(checkpoint).with_suffix("")  # strip .zip if present
    sibling = ckpt.parent / ckpt.name.replace("ppo_", "ppo_vecnormalize_", 1)
    candidates = [
        sibling.with_suffix(".pkl"),
        ckpt.parent / "vec_normalize.pkl",
        ckpt.parent.parent / "vec_normalize.pkl",
    ]
    for c in candidates:
        remote_path = f"{remote_dir}/{c}"
        if _remote_file_exists(ssh_host, remote_path):
            return str(c)
    raise FileNotFoundError(
        f"No VecNormalize stats found on remote for checkpoint {checkpoint}. "
        f"Searched: {[str(c) for c in candidates]}"
    )


def _scp(ssh_host: str, remote_path: str, local_path: Path):
    """SCP a single file from the remote machine."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"scp {ssh_host}:{remote_path} -> {local_path}")
    subprocess.run(
        ["scp", f"{ssh_host}:{remote_path}", str(local_path)],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Fetch a checkpoint from a remote machine and visualize with enjoy.py",
    )
    parser.add_argument("ssh_host", help="SSH host, e.g. danyoungday@1.2.3.4")
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default="logs/ppo_final",
        help="Checkpoint path relative to project root (default: logs/ppo_final)",
    )
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes (default: 10)")
    parser.add_argument(
        "--remote-dir",
        default="~/workspace/robteach",
        help="Remote project root (default: ~/workspace/robteach)",
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint
    remote_dir = args.remote_dir

    # 1. Resolve remote files
    model_rel = str(Path(checkpoint).with_suffix("")) + ".zip"
    print(f"Looking for model: {remote_dir}/{model_rel}")
    if not _remote_file_exists(args.ssh_host, f"{remote_dir}/{model_rel}"):
        sys.exit(f"Error: remote model not found at {remote_dir}/{model_rel}")

    print("Looking for VecNormalize stats...")
    norm_rel = _find_remote_vec_normalize(args.ssh_host, remote_dir, checkpoint)

    # 2. SCP both files to matching local paths
    local_model = PROJECT_ROOT / model_rel
    local_norm = PROJECT_ROOT / norm_rel

    _scp(args.ssh_host, f"{remote_dir}/{model_rel}", local_model)
    _scp(args.ssh_host, f"{remote_dir}/{norm_rel}", local_norm)

    # 3. Run enjoy.py
    local_checkpoint = str(PROJECT_ROOT / checkpoint)
    cmd = ["mjpython", "enjoy.py", local_checkpoint, "--episodes", str(args.episodes)]
    print(f"Running: {' '.join(cmd)}")
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
