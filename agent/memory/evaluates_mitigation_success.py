"""
src/memory/memory_reflection.py

Memory & Reflection System
- Logs past disruptions (disruption events + context)
- Evaluates mitigation success (actual vs expected, service recovery, cost, timeline)
- Improves future recommendations (updates supplier risk priors, action effectiveness, thresholds)

Storage:
- JSONL event log at: data/memory/events.jsonl
- JSON snapshots of aggregated stats at: data/memory/state.json

This module is designed to plug into:
- PlanningEngine (after scenarios / decisions)
- ActionExecutor (after execution results)
- ERPIntegrator (after PO updates / inventory updates)

Works with async patterns, but uses simple file IO.
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import logging

logger = logging.getLogger(__name__)


# -----------------------------
# Data Models (lightweight)
# -----------------------------

@dataclass
class DisruptionEvent:
    event_id: str
    company_id: str
    timestamp: str
    disruption_type: str
    severity_score: float
    affected_regions: List[str]
    description: str
    observed_impacts: Dict[str, Any]  # e.g., {"late_shipments": 120, "stockout_hours": 6}
    context: Dict[str, Any]           # e.g., {"supplier_ids": [...], "routes": [...], "signals": {...}}


@dataclass
class MitigationExecutionEvent:
    event_id: str
    company_id: str
    timestamp: str
    action_id: str
    action_type: str
    priority_level: str
    execution_mode: str             # automatic / human_approval / executive_approval
    status: str                     # completed / failed / waiting_approval
    expected: Dict[str, Any]        # expected outcomes (benefit, risk reduction, etc.)
    actual: Dict[str, Any]          # actual outcomes (cost, lead time, etc.)
    metadata: Dict[str, Any]        # any extra info (emails_sent, workflows, erp_updates)


@dataclass
class ReflectionUpdate:
    event_id: str
    company_id: str
    timestamp: str
    update_type: str               # "supplier_prior", "action_effectiveness", "threshold_tuning"
    payload: Dict[str, Any]


# -----------------------------
# Persistent Event Store
# -----------------------------

class MemoryStore:
    """
    Simple JSONL event store:
    - append-only events.jsonl
    - aggregated state.json snapshot for fast startup
    """

    def __init__(self, base_dir: str = "data/memory"):
        self.base_dir = base_dir
        self.events_path = os.path.join(base_dir, "events.jsonl")
        self.state_path = os.path.join(base_dir, "state.json")

        os.makedirs(self.base_dir, exist_ok=True)

        # Aggregated state in memory (loaded at startup)
        self.state: Dict[str, Any] = {
            "schema_version": 1,
            "companies": {},  # company_id -> company_state
        }

        self._loaded = False

    async def load(self) -> None:
        """Load snapshot state (and optionally replay events if missing)."""
        if self._loaded:
            return

        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    self.state = json.load(f)
                logger.info("MemoryStore state loaded from snapshot.")
            except Exception as e:
                logger.warning(f"Failed to load state snapshot: {e}. Rebuilding from events.")
                await self._rebuild_from_events()
        else:
            # No snapshot exists; rebuild from events if they exist
            await self._rebuild_from_events()

        self._loaded = True

    async def append_event(self, event: Dict[str, Any]) -> None:
        """Append an event to JSONL log."""
        await self.load()
        line = json.dumps(event, ensure_ascii=False)

        # file append is fast; still wrap in async-friendly call
        def _write():
            with open(self.events_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

        await asyncio.to_thread(_write)

    async def save_snapshot(self) -> None:
        """Write state snapshot to disk."""
        await self.load()

        def _write():
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=2, ensure_ascii=False)

        await asyncio.to_thread(_write)

    async def _rebuild_from_events(self) -> None:
        """Rebuild state by replaying events.jsonl."""
        self.state = {"schema_version": 1, "companies": {}}

        if not os.path.exists(self.events_path):
            logger.info("No events log found; starting fresh.")
            return

        def _read_lines() -> List[str]:
            with open(self.events_path, "r", encoding="utf-8") as f:
                return f.read().splitlines()

        lines = await asyncio.to_thread(_read_lines)

        for line in lines:
            if not line.strip():
                continue
            try:
                ev = json.loads(line)
                await self._apply_event_to_state(ev)
            except Exception as e:
                logger.warning(f"Skipping corrupted event line: {e}")

        logger.info("State rebuilt from events log.")
        await self.save_snapshot()

    async def _apply_event_to_state(self, ev: Dict[str, Any]) -> None:
        """Apply one event to in-memory aggregated state."""
        company_id = str(ev.get("company_id", "unknown"))
        cstate = self.state["companies"].setdefault(company_id, _default_company_state())

        etype = ev.get("event_type")

        if etype == "disruption":
            cstate["disruptions"]["count"] += 1
            cstate["disruptions"]["by_type"][ev.get("disruption_type", "unknown")] = \
                cstate["disruptions"]["by_type"].get(ev.get("disruption_type", "unknown"), 0) + 1
            cstate["disruptions"]["severity_history"].append(float(ev.get("severity_score", 0.0)))

        elif etype == "mitigation_execution":
            cstate["actions"]["count"] += 1
            at = ev.get("action_type", "unknown")
            cstate["actions"]["by_type"][at] = cstate["actions"]["by_type"].get(at, 0) + 1

            status = ev.get("status", "unknown")
            cstate["actions"]["by_status"][status] = cstate["actions"]["by_status"].get(status, 0) + 1

            # Evaluate success signal if available
            eval_block = ev.get("evaluation", {})
            if eval_block:
                cstate["actions"]["success_scores"].append(float(eval_block.get("success_score", 0.0)))
                cstate["actions"]["roi_realized"].append(float(eval_block.get("roi_realized", 0.0)))
                cstate["actions"]["service_impact"].append(float(eval_block.get("service_impact", 0.0)))

                # Track action effectiveness per type
                eff = cstate["action_effectiveness"].setdefault(at, _default_action_eff())
                eff["n"] += 1
                eff["success_scores"].append(float(eval_block.get("success_score", 0.0)))
                eff["roi_realized"].append(float(eval_block.get("roi_realized", 0.0)))
                eff["time_to_effect_days"].append(float(eval_block.get("time_to_effect_days", 0.0)))

        elif etype == "reflection_update":
            # reflection updates are already aggregated; keep last updates list
            cstate["reflection"]["updates"].append(ev.get("payload", {}))

        # Keep limited history
        _cap_list(cstate["disruptions"]["severity_history"], 500)
        _cap_list(cstate["actions"]["success_scores"], 500)
        _cap_list(cstate["actions"]["roi_realized"], 500)
        _cap_list(cstate["actions"]["service_impact"], 500)
        _cap_list(cstate["reflection"]["updates"], 200)


def _default_company_state() -> Dict[str, Any]:
    return {
        "disruptions": {
            "count": 0,
            "by_type": {},
            "severity_history": [],
        },
        "actions": {
            "count": 0,
            "by_type": {},
            "by_status": {},
            "success_scores": [],
            "roi_realized": [],
            "service_impact": [],
        },
        "action_effectiveness": {},  # action_type -> {n, success_scores[], roi_realized[], time_to_effect_days[]}
        "supplier_priors": {},        # supplier_id -> prior risk score (0-1)
        "policy_tuning": {
            "po_auto_cost_limit": 100000.0,
            "auto_execution_threshold": 0.80,
            "human_approval_threshold": 0.50,
        },
        "reflection": {
            "updates": [],
            "last_reflection_at": None,
        },
    }


def _default_action_eff() -> Dict[str, Any]:
    return {"n": 0, "success_scores": [], "roi_realized": [], "time_to_effect_days": []}


def _cap_list(lst: List[Any], cap: int) -> None:
    if len(lst) > cap:
        del lst[0:len(lst) - cap]


# -----------------------------
# Reflection Engine
# -----------------------------

class ReflectionEngine:
    """
    Computes:
    - mitigation success score
    - updates to supplier priors
    - action effectiveness stats
    - threshold tuning suggestions
    """

    def __init__(self, store: MemoryStore):
        self.store = store

    async def log_disruption(self, disruption: DisruptionEvent) -> Dict[str, Any]:
        """Log a disruption event (past disruption memory)."""
        ev = asdict(disruption)
        ev["event_type"] = "disruption"
        await self.store.append_event(ev)
        await self.store._apply_event_to_state(ev)
        await self.store.save_snapshot()
        return {"logged": True, "event_id": disruption.event_id}

    async def log_mitigation_execution(
        self,
        exec_event: MitigationExecutionEvent,
        supplier_ids_involved: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Log execution + compute evaluation signal.
        Supplier IDs can be passed to update priors.
        """
        supplier_ids_involved = supplier_ids_involved or []

        evaluation = self.evaluate_mitigation_success(exec_event.expected, exec_event.actual)
        ev = asdict(exec_event)
        ev["event_type"] = "mitigation_execution"
        ev["evaluation"] = evaluation

        await self.store.append_event(ev)
        await self.store._apply_event_to_state(ev)

        # Reflection: update supplier priors + action effectiveness + policy tuning
        updates: List[ReflectionUpdate] = []
        updates.extend(await self._update_supplier_priors(exec_event.company_id, supplier_ids_involved, evaluation))
        updates.extend(await self._update_action_effectiveness(exec_event.company_id, exec_event.action_type, evaluation))
        updates.extend(await self._maybe_tune_policy(exec_event.company_id))

        for u in updates:
            uev = asdict(u)
            uev["event_type"] = "reflection_update"
            await self.store.append_event(uev)
            await self.store._apply_event_to_state(uev)

        # mark reflection time
        cstate = self.store.state["companies"].setdefault(str(exec_event.company_id), _default_company_state())
        cstate["reflection"]["last_reflection_at"] = datetime.utcnow().isoformat()

        await self.store.save_snapshot()

        return {
            "logged": True,
            "event_id": exec_event.event_id,
            "evaluation": evaluation,
            "reflection_updates": [asdict(u) for u in updates],
        }

    # -----------------------------
    # Evaluation logic
    # -----------------------------

    def evaluate_mitigation_success(self, expected: Dict[str, Any], actual: Dict[str, Any]) -> Dict[str, Any]:
        """
        Produce a success signal in [0,1].

        expected can include:
        - expected_benefit, expected_cost, expected_risk_reduction, expected_time_days
        actual can include:
        - actual_benefit, actual_cost, actual_risk_reduction, time_to_effect_days, service_impact (-1..+1)
        """
        exp_cost = float(expected.get("expected_cost", expected.get("cost", 0.0)) or 0.0)
        exp_benefit = float(expected.get("expected_benefit", expected.get("benefit", 0.0)) or 0.0)
        exp_rr = float(expected.get("expected_risk_reduction", 0.0) or 0.0)
        exp_time = float(expected.get("expected_time_days", expected.get("time_days", 0.0)) or 0.0)

        act_cost = float(actual.get("actual_cost", actual.get("cost", exp_cost)) or 0.0)
        act_benefit = float(actual.get("actual_benefit", actual.get("benefit", exp_benefit)) or 0.0)
        act_rr = float(actual.get("actual_risk_reduction", actual.get("risk_reduction", exp_rr)) or 0.0)
        tte = float(actual.get("time_to_effect_days", actual.get("time_days", exp_time)) or (exp_time or 0.0))

        # service_impact: positive is good (improved service), negative is bad (stockouts)
        service_impact = float(actual.get("service_impact", 0.0) or 0.0)

        # ROI realized (benefit/cost)
        roi_realized = (act_benefit / act_cost) if act_cost > 0 else 0.0

        # Score components
        rr_score = _safe_ratio(act_rr, max(exp_rr, 1e-6))  # >1 means exceeded
        rr_score = min(rr_score, 1.5) / 1.5               # clamp to [0,1]

        roi_score = min(max(roi_realized, 0.0), 3.0) / 3.0

        # time score: faster is better
        if exp_time > 0:
            time_ratio = exp_time / max(tte, 1e-6)
            time_score = min(max(time_ratio, 0.0), 2.0) / 2.0
        else:
            time_score = 0.5  # unknown

        # service score: map roughly -1..+1 to 0..1
        service_score = max(0.0, min(1.0, 0.5 + 0.5 * service_impact))

        # Weighted overall success
        success_score = (
            0.35 * rr_score +
            0.35 * roi_score +
            0.20 * time_score +
            0.10 * service_score
        )

        return {
            "success_score": float(max(0.0, min(1.0, success_score))),
            "roi_realized": float(roi_realized),
            "risk_reduction_realized": float(act_rr),
            "time_to_effect_days": float(tte),
            "service_impact": float(service_impact),
            "components": {
                "rr_score": rr_score,
                "roi_score": roi_score,
                "time_score": time_score,
                "service_score": service_score,
            },
        }

    # -----------------------------
    # Improvement updates
    # -----------------------------

    async def _update_supplier_priors(
        self,
        company_id: str,
        supplier_ids: List[str],
        evaluation: Dict[str, Any]
    ) -> List[ReflectionUpdate]:
        """
        Update supplier prior risk scores based on whether mitigations involving them succeed/fail.
        Simple Bayesian-ish heuristic:
        - good outcomes reduce risk prior slightly
        - bad outcomes increase risk prior slightly
        """
        if not supplier_ids:
            return []

        cstate = self.store.state["companies"].setdefault(str(company_id), _default_company_state())
        priors = cstate["supplier_priors"]

        success = float(evaluation.get("success_score", 0.5))

        updates = []
        for sid in supplier_ids:
            old = float(priors.get(sid, 0.5))
            # learning rate small and stable
            lr = 0.05

            # if success high → reduce risk; if success low → increase risk
            delta = (0.5 - success) * lr  # success>0.5 => negative delta
            new = max(0.0, min(1.0, old + delta))

            priors[sid] = new
            updates.append(ReflectionUpdate(
                event_id=_eid("ref_supplier"),
                company_id=str(company_id),
                timestamp=datetime.utcnow().isoformat(),
                update_type="supplier_prior",
                payload={"supplier_id": sid, "old_prior": old, "new_prior": new, "success_score": success},
            ))
        return updates

    async def _update_action_effectiveness(
        self,
        company_id: str,
        action_type: str,
        evaluation: Dict[str, Any]
    ) -> List[ReflectionUpdate]:
        """
        Maintain an effectiveness score per action type.
        This is what you later feed into PlanningEngine to prefer actions that historically worked.
        """
        cstate = self.store.state["companies"].setdefault(str(company_id), _default_company_state())
        eff = cstate["action_effectiveness"].setdefault(action_type, _default_action_eff())

        eff["n"] += 1
        eff["success_scores"].append(float(evaluation.get("success_score", 0.0)))
        eff["roi_realized"].append(float(evaluation.get("roi_realized", 0.0)))
        eff["time_to_effect_days"].append(float(evaluation.get("time_to_effect_days", 0.0)))

        _cap_list(eff["success_scores"], 300)
        _cap_list(eff["roi_realized"], 300)
        _cap_list(eff["time_to_effect_days"], 300)

        summary = _summarize_effectiveness(eff)

        return [ReflectionUpdate(
            event_id=_eid("ref_action"),
            company_id=str(company_id),
            timestamp=datetime.utcnow().isoformat(),
            update_type="action_effectiveness",
            payload={"action_type": action_type, "summary": summary},
        )]

    async def _maybe_tune_policy(self, company_id: str) -> List[ReflectionUpdate]:
        """
        Tune execution/PO thresholds based on history.
        Example policies:
        - If many auto actions fail → raise auto threshold (be more conservative)
        - If many actions succeed → lower auto threshold slightly
        - If PO adjustments frequently exceed limit but succeed → raise PO auto limit slightly (optional)
        """
        cstate = self.store.state["companies"].setdefault(str(company_id), _default_company_state())

        success_scores = cstate["actions"]["success_scores"]
        if len(success_scores) < 20:
            return []  # not enough data yet

        avg_success = float(statistics.mean(success_scores[-20:]))

        policy = cstate["policy_tuning"]
        auto_th = float(policy.get("auto_execution_threshold", 0.80))

        # conservative tuning band
        if avg_success < 0.45:
            auto_th = min(0.92, auto_th + 0.03)
        elif avg_success > 0.70:
            auto_th = max(0.70, auto_th - 0.02)

        policy["auto_execution_threshold"] = auto_th

        return [ReflectionUpdate(
            event_id=_eid("ref_policy"),
            company_id=str(company_id),
            timestamp=datetime.utcnow().isoformat(),
            update_type="threshold_tuning",
            payload={"auto_execution_threshold": auto_th, "avg_success_last_20": avg_success},
        )]

    # -----------------------------
    # Query API (for PlanningEngine)
    # -----------------------------

    async def get_company_memory_profile(self, company_id: str) -> Dict[str, Any]:
        """Returns aggregated state used to improve planning decisions."""
        await self.store.load()
        cstate = self.store.state["companies"].get(str(company_id), _default_company_state())

        # Precompute useful “policy hints”
        action_rankings = self.rank_actions_by_effectiveness(company_id)

        return {
            "company_id": str(company_id),
            "disruption_summary": cstate["disruptions"],
            "action_summary": cstate["actions"],
            "supplier_priors": cstate["supplier_priors"],
            "policy_tuning": cstate["policy_tuning"],
            "action_effectiveness_ranked": action_rankings,
            "last_reflection_at": cstate["reflection"]["last_reflection_at"],
        }

    def rank_actions_by_effectiveness(self, company_id: str) -> List[Dict[str, Any]]:
        """Rank action types by success_score and ROI (simple composite)."""
        cstate = self.store.state["companies"].get(str(company_id), _default_company_state())
        out = []

        for atype, eff in cstate["action_effectiveness"].items():
            summ = _summarize_effectiveness(eff)
            score = 0.6 * summ["avg_success"] + 0.4 * summ["avg_roi_norm"]
            out.append({"action_type": atype, "score": score, "summary": summ})

        out.sort(key=lambda x: x["score"], reverse=True)
        return out


# -----------------------------
# Utility Functions
# -----------------------------

def _safe_ratio(a: float, b: float) -> float:
    if b <= 0:
        return 1.0
    return a / b

def _eid(prefix: str) -> str:
    return f"{prefix}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"

def _summarize_effectiveness(eff: Dict[str, Any]) -> Dict[str, Any]:
    def _avg(x: List[float]) -> float:
        return float(statistics.mean(x)) if x else 0.0

    avg_success = _avg(eff.get("success_scores", []))
    avg_roi = _avg(eff.get("roi_realized", []))
    avg_tte = _avg(eff.get("time_to_effect_days", []))

    # Normalize ROI into [0,1] for comparisons (cap at 3x)
    avg_roi_norm = min(max(avg_roi, 0.0), 3.0) / 3.0

    return {
        "n": int(eff.get("n", 0)),
        "avg_success": avg_success,
        "avg_roi": avg_roi,
        "avg_roi_norm": avg_roi_norm,
        "avg_time_to_effect_days": avg_tte,
    }