from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from org_agent.api import lookup_organization
from org_agent.models import PROFILE_DISPLAY_FIELDS


MISSING_GROUND_TRUTH = "(not in ground truth)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare agent output against JSONL ground truth.")
    parser.add_argument("ground_truth", help="JSONL file with name, optional website, and expected fields")
    parser.add_argument("index", nargs="?", type=int, help="Optional zero-based row index to evaluate")
    parser.add_argument(
        "--config",
        help="Optional registry YAML config with endpoints to query before extraction.",
    )
    parser.add_argument(
        "--registry",
        action="append",
        default=[],
        dest="registries",
        help="Enable optional registry provider(s), e.g. zefix. Repeatable.",
    )
    args = parser.parse_args()

    rows = _read_jsonl(Path(args.ground_truth))
    if args.index is not None:
        if args.index < 0 or args.index >= len(rows):
            raise SystemExit(f"Index {args.index} out of range. Valid range: 0-{len(rows) - 1}")
        rows = [rows[args.index]]
    console = Console()

    for row in tqdm(rows, desc="Evaluating", unit="case"):
        name = row["name"]
        website = row.get("website")
        expected = row.get("expected", {})

        console.rule(name)
        try:
            profile = lookup_organization(
                name=name,
                website=website,
                config=args.config,
                registries=args.registries,
                progress=None,
            )
        except Exception as exc:  # noqa: BLE001 - keep batch evaluation running
            console.print(f"[red]Agent failed:[/red] {exc}")
            continue

        prediction = profile.model_dump()
        table = Table(show_header=True)
        table.add_column("Field")
        table.add_column("Ground Truth")
        table.add_column("Agent")

        for field in PROFILE_DISPLAY_FIELDS:
            expected_value = expected.get(field, MISSING_GROUND_TRUTH)
            actual_value = prediction.get(field)
            table.add_row(field, _format_value(expected_value), _format_value(actual_value))

        console.print(table)


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _format_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


if __name__ == "__main__":
    main()
