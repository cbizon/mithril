"""Command-line entry point: ``biolink-rules <stage> ...``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from biolink_rules.ingest import ingest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="biolink-rules")
    stages = parser.add_subparsers(dest="stage", required=True)

    p_ingest = stages.add_parser("ingest", help="KGX jsonl -> compact arrays (once per release)")
    p_ingest.add_argument("--kg", type=Path, required=True, help="Directory with nodes.jsonl and edges.jsonl")
    p_ingest.add_argument("--out", type=Path, required=True, help="Output directory")

    args = parser.parse_args(argv)
    if args.stage == "ingest":
        print(json.dumps(ingest(args.kg, args.out), indent=2))
