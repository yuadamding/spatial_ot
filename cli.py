from __future__ import annotations

import argparse
import json

from .config import load_config
from .training import run_experiment
from .visualization import plot_preprocessed_inputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Spatial OT teacher-student niche model.")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Run a staged training experiment.")
    train.add_argument("--config", required=True, help="Path to a TOML config file.")

    plot_inputs = sub.add_parser("plot-inputs", help="Render a 2D overview of the preprocessed input data.")
    plot_inputs.add_argument("--config", required=True, help="Path to a TOML config file.")
    plot_inputs.add_argument("--output", help="Optional output PNG path.")
    plot_inputs.add_argument("--cell-subset", type=int, help="Override cell subset size. Use 0 for all cells.")
    plot_inputs.add_argument("--bin-subset", type=int, help="Override bin subset size. Use 0 for all bins.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train":
        config = load_config(args.config)
        summary = run_experiment(config)
        print(json.dumps(summary, indent=2))
    elif args.command == "plot-inputs":
        config = load_config(args.config)
        output_path = plot_preprocessed_inputs(
            config=config,
            cell_subset=args.cell_subset,
            bin_subset=args.bin_subset,
            output_path=args.output,
        )
        print(json.dumps({"output_path": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
