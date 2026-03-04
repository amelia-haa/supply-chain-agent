from __future__ import annotations

import argparse
import json

from agent.perception.live_ingest_stub import write_live_signals


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch live disruption signals and write normalized JSON")
    parser.add_argument("--max-items", type=int, default=30, help="Maximum normalized disruption signals to keep")
    parser.add_argument(
        "--output",
        type=str,
        default="data/live_disruption_signals.json",
        help="Output JSON path",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = write_live_signals(output_path=args.output, max_items=max(1, args.max_items))
    print(json.dumps(payload.get("meta", {}), indent=2))


if __name__ == "__main__":
    main()
