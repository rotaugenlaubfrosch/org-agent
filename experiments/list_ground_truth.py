from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="List names from a JSONL ground truth file.")
    parser.add_argument(
        "ground_truth",
        nargs="?",
        default=Path(__file__).with_name("ground_truth.jsonl"),
        type=Path,
        help="JSONL file with a top-level name field. Defaults to experiments/ground_truth.jsonl.",
    )
    args = parser.parse_args()

    for index, row in enumerate(_read_jsonl(args.ground_truth)):
        print(f"{index}: {row.get('name', '')}")


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    main()
