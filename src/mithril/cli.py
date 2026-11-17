"""Command-line entry point: ``mithril <stage> ...``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mithril.build import build
from mithril.ingest import ingest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mithril")
    stages = parser.add_subparsers(dest="stage", required=True)

    p_ingest = stages.add_parser("ingest", help="KGX jsonl -> compact arrays (once per release)")
    p_ingest.add_argument("--kg", type=Path, required=True, help="Directory with nodes.jsonl and edges.jsonl")
    p_ingest.add_argument("--out", type=Path, required=True, help="Output directory")

    p_build = stages.add_parser("build", help="ingest + config -> deduplicated edges per semantic atom")
    p_build.add_argument("--ingest", type=Path, required=True, help="Ingest output directory")
    p_build.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    p_build.add_argument("--out", type=Path, required=True, help="Output directory")

    args = parser.parse_args(argv)
    if args.stage == "ingest":
        print(json.dumps(ingest(args.kg, args.out), indent=2))
    elif args.stage == "build":
        print(json.dumps(build(args.ingest, args.config, args.out), indent=2))
