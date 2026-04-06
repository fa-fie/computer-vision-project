from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from generator import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate physical attack images.")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("config.yaml")),
        help="Path to a YAML config file.",
    )
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    summary = run_pipeline(config, config_path=Path(args.config))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
