"""SCP a curriculum stage from a remote Linux machine and visualize locally.

Fetches the best model, VecNormalize stats, curriculum env, and config,
then execs enjoy.py via mjpython.

Usage:
    mjpython fetch_and_enjoy.py user@host results/morehist/stage_2
    mjpython fetch_and_enjoy.py user@host results/morehist/stage_2 --episodes 5
    mjpython fetch_and_enjoy.py user@host results/morehist/stage_2 --remote-dir ~/other/project
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
        description="Fetch a curriculum stage from a remote machine and visualize with enjoy.py",
    )
    parser.add_argument("ssh_host", help="SSH host, e.g. user@1.2.3.4")
    parser.add_argument(
        "stage_dir",
        help="Stage directory relative to remote project root, e.g. results/morehist/stage_2",
    )
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes (default: 10)")
    parser.add_argument(
        "--remote-dir",
        default="~/workspace/robteach",
        help="Remote project root (default: ~/workspace/robteach)",
    )
    args = parser.parse_args()

    remote_root = args.remote_dir
    stage = args.stage_dir.rstrip("/")

    # Deterministic remote paths relative to stage_dir
    files = {
        "model": f"{stage}/best/best_model.zip",
        "vec_normalize": f"{stage}/best/vec_normalize.pkl",
        "curriculum_env": f"{stage}/curriculum_env.py",
        "config": str(Path(stage).parent / "base_config.yaml"),
    }

    # Verify all remote files exist
    for name, rel_path in files.items():
        full = f"{remote_root}/{rel_path}"
        print(f"Checking {name}: {full}")
        if not _remote_file_exists(args.ssh_host, full):
            sys.exit(f"Error: {name} not found at {full}")

    # SCP all files to matching local paths
    local_paths = {}
    for name, rel_path in files.items():
        local = PROJECT_ROOT / rel_path
        _scp(args.ssh_host, f"{remote_root}/{rel_path}", local)
        local_paths[name] = local

    # Run enjoy.py
    cmd = [
        "mjpython", "enjoy.py",
        str(local_paths["model"]),
        "--episodes", str(args.episodes),
        "--config", str(local_paths["config"]),
        "--env-cls-path", str(local_paths["curriculum_env"]),
    ]
    print(f"Running: {' '.join(cmd)}")
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
