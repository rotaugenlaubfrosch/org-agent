from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from org_agent.api import lookup_organization
from org_agent.models import profile_display_field_groups


MISSING_GROUND_TRUTH = "(not in ground truth)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare agent output against JSONL ground truth.")
    parser.add_argument("ground_truth", help="JSONL file with name, website, and expected fields")
    parser.add_argument("index", nargs="?", type=int, help="Optional zero-based row index to evaluate")
    parser.add_argument(
        "--country",
        help="Enable optional country registry integration by country code, e.g. ch.",
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
        if not website:
            raise ValueError(f"Ground-truth row for {name!r} is missing required website.")
        expected = row.get("expected", {})
        expected = {"queried_name": name, **expected}

        console.rule(name)
        try:
            result = lookup_organization(
                name=name,
                website=website,
                country=args.country,
                progress=None,
            )
        except Exception as exc:  # noqa: BLE001 - keep batch evaluation running
            console.print(f"[red]Agent failed:[/red] {exc}")
            continue

        website_prediction = result.website_profile.model_dump()
        registry_prediction = (
            result.registry_profile.model_dump() if result.registry_profile is not None else {}
        )
        table = Table(show_header=True)
        table.add_column("Field")
        table.add_column("Ground Truth")
        table.add_column("Agent")

        website_fields, registry_fields = profile_display_field_groups()
        expected_address_fields = expected.get("address_fields", {})
        for field in website_fields:
            expected_value = expected.get(field, MISSING_GROUND_TRUTH)
            actual_value = website_prediction.get(field)
            table.add_row(field, _format_value(expected_value), _format_value(actual_value))
            if field == "queried_country":
                table.add_section()
            if field == "address":
                for address_field, address_value in result.website_profile.address_fields.items():
                    expected_address_value = expected_address_fields.get(
                        address_field,
                        MISSING_GROUND_TRUTH,
                    )
                    table.add_row(
                        f"  {address_field}",
                        _format_value(expected_address_value),
                        _format_value(address_value),
                    )
        table.add_section()
        for field in registry_fields:
            expected_value = expected.get(field, MISSING_GROUND_TRUTH)
            actual_value = registry_prediction.get(field)
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
