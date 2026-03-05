from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import FastAPI
from pydantic import BaseModel, Field

from agent.perception.live_ingest_stub import write_live_signals
from agent.tools import (
    get_company_profile,
    run_cost_optimized_pipeline,
    run_full_cycle,
)


app = FastAPI(title="Supply Chain Agent Cloud Runner", version="1.0.0")


class RunCycleRequest(BaseModel):
    company_ids: List[str] = Field(default_factory=lambda: ["de_semiconductor_auto", "mx_multisource_industrial"])
    max_items: int = 40
    force_run: bool = False
    include_full_output: bool = False


def _extract_retry_after_seconds(error_text: str) -> int:
    text = str(error_text or "")
    patterns = [
        r"retry in ([0-9]+(?:\.[0-9]+)?)s",
        r"retryDelay['\"]?:\s*'([0-9]+)s'",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            try:
                return max(1, int(float(m.group(1))))
            except ValueError:
                continue
    return 60


def _check_new_signals(company_ids: List[str]) -> Dict[str, Any]:
    """
    Cheap deterministic gate:
    - Run staged pipeline
    - Read event_driven_skip flag per company
    - Trigger full cycle only if at least one company has new/changed events
    """
    checks: List[Dict[str, Any]] = []
    should_run = False
    for cid in company_ids:
        company = get_company_profile(cid)
        pipeline = run_cost_optimized_pipeline(company)
        stats = pipeline.get("pipeline_stats", {})
        skip = bool(stats.get("event_driven_skip", False))
        if not skip:
            should_run = True
        checks.append(
            {
                "company_id": cid,
                "event_driven_skip": skip,
                "new_or_changed_events": int(stats.get("event_driven_new_or_changed", 0)),
                "raw_signals": int(stats.get("raw_signals", 0)),
            }
        )
    return {"should_run": should_run, "checks": checks}


@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    return {
        "ok": True,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_mode": os.getenv("APP_RUNTIME_MODE", "api"),
        "signal_source": os.getenv("APP_SIGNAL_SOURCE", "live"),
    }


@app.post("/run-cycle")
def run_cycle(req: RunCycleRequest) -> Dict[str, Any]:
    started = datetime.now(timezone.utc).isoformat()

    # 1) Ingest latest live signals.
    ingest_meta = write_live_signals(output_path="data/live_disruption_signals.json", max_items=max(1, req.max_items)).get("meta", {})

    # 2) Cheap event-driven gate.
    gate = _check_new_signals(req.company_ids)
    if (not req.force_run) and (not gate["should_run"]):
        return {
            "status": "skipped_no_new_signals",
            "started_utc": started,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "ingest_meta": ingest_meta,
            "gate": gate,
            "llm_calls_saved": True,
        }

    # 3) Only now run full cycle.
    try:
        result = run_full_cycle(
            company_ids=",".join(req.company_ids),
            include_full_output=req.include_full_output,
        )
    except Exception as exc:
        err = str(exc)
        resource_exhausted = ("429" in err) or ("RESOURCE_EXHAUSTED" in err)
        if resource_exhausted:
            retry_after = _extract_retry_after_seconds(err)
            return {
                "status": "deferred_quota",
                "started_utc": started,
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "ingest_meta": ingest_meta,
                "gate": gate,
                "retry_after_seconds": retry_after,
                "message": (
                    "LLM quota was temporarily exhausted. Please retry after the suggested delay. "
                    "Deterministic pre-filtering is still active."
                ),
                "error": err[:500],
            }
        raise
    return {
        "status": "processed",
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "ingest_meta": ingest_meta,
        "gate": gate,
        "result": result,
    }
