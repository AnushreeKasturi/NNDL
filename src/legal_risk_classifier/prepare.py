from __future__ import annotations

import argparse
from pathlib import Path

from .data import convert_cuad_to_rows, save_rows_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert CUAD JSON into JSONL rows for training.")
    parser.add_argument("--cuad_json", type=Path, required=True, help="Path to CUAD_v1.json")
    parser.add_argument("--output_jsonl", type=Path, required=True, help="Output JSONL path")
    parser.add_argument(
        "--drop_unlabeled",
        action="store_true",
        help="Drop paragraphs without any of the 6 selected labels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = convert_cuad_to_rows(args.cuad_json, include_unlabeled=not args.drop_unlabeled)
    save_rows_jsonl(rows, args.output_jsonl)
    print(f"Wrote {len(rows)} rows to {args.output_jsonl}")


if __name__ == "__main__":
    main()

