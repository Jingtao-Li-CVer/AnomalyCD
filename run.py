"""AnomalyCD full pipeline: stage 1 -> stage 2."""

from __future__ import annotations

import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AnomalyCD stage 1 and stage 2 sequentially"
    )
    parser.add_argument(
        "--stage",
        choices=["all", "1", "2"],
        default="all",
        help="Pipeline stage to run",
    )
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--stage1-root", default=None)
    parser.add_argument("--stage2-root", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--event", default=None, help="Process a single event directory")
    parser.add_argument(
        "--no-skip-existing-stage1",
        action="store_true",
        help="Re-run stage 1 even when outputs already exist",
    )
    return parser.parse_args()


def build_command(script_name: str, args: argparse.Namespace) -> list[str]:
    command = [sys.executable, script_name]
    if args.data_root:
        command.extend(["--data-root", args.data_root])
    if args.device:
        command.extend(["--device", args.device])
    if args.event:
        command.extend(["--event", args.event])

    if script_name == "run_stage1.py":
        if args.stage1_root:
            command.extend(["--output-root", args.stage1_root])
        if args.no_skip_existing_stage1:
            command.append("--no-skip-existing")
    elif script_name == "run_stage2.py":
        if args.stage1_root:
            command.extend(["--stage1-root", args.stage1_root])
        if args.stage2_root:
            command.extend(["--output-root", args.stage2_root])

    return command


def main() -> None:
    args = parse_args()

    if args.stage in ("all", "1"):
        subprocess.run(build_command("run_stage1.py", args), check=True)

    if args.stage in ("all", "2"):
        subprocess.run(build_command("run_stage2.py", args), check=True)


if __name__ == "__main__":
    main()
