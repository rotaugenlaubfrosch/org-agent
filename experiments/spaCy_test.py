from __future__ import annotations

import argparse
from pathlib import Path

import spacy


def main() -> None:
    parser = argparse.ArgumentParser(description="Run spaCy NER over crawl log text files.")
    parser.add_argument("log_dir", help="Directory containing crawl log .txt files")
    parser.add_argument("--model", default="en_core_web_sm", help="spaCy model to use")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.is_dir():
        raise SystemExit(f"Log directory not found: {log_dir}")

    try:
        nlp = spacy.load(args.model)
    except OSError as exc:
        raise SystemExit(
            f"Could not load spaCy model '{args.model}'. Install it with:\n"
            f"python -m spacy download {args.model}"
        ) from exc

    files = sorted(log_dir.rglob("*.txt"))
    if not files:
        raise SystemExit(f"No .txt files found in: {log_dir}")

    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        doc = nlp(text)

        print(f"\n=== {path} ===")
        if not doc.ents:
            print("No entities found.")
            continue

        for ent in doc.ents:
            print(f"{ent.label_:<12} {ent.text}")


if __name__ == "__main__":
    main()
