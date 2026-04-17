from __future__ import annotations

import argparse
import json

from .config import load_config
from .training import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Spatial OT teacher-student niche model.")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Run a staged training experiment.")
    train.add_argument("--config", required=True, help="Path to a TOML config file.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train":
        config = load_config(args.config)
        summary = run_experiment(config)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
