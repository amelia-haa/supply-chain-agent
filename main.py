from __future__ import annotations

import argparse
import asyncio
import json

from agent.autonomous_loop import run_autonomous_loop


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Autonomous Supply Chain Resilience Agent Demo")
    parser.add_argument("--cycles", type=int, default=1, help="Number of autonomous cycles to run")
    parser.add_argument("--interval-seconds", type=int, default=10, help="Sleep interval between cycles")
    parser.add_argument(
        "--companies",
        type=str,
        default="de_semiconductor_auto",
        help="Comma-separated company profile IDs from data/company_profiles.json",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    outputs = asyncio.run(
        run_autonomous_loop(
            cycles=max(1, args.cycles),
            interval_seconds=max(1, args.interval_seconds),
            company_ids=[s.strip() for s in args.companies.split(",") if s.strip()],
        )
    )
    print(json.dumps({"runs": outputs}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
