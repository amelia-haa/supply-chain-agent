# src/memory/reflection_engine.py
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional

DEFAULT_EVENTS_PATH = "data/memory/events.jsonl"
DEFAULT_SNAPSHOTS_PATH = "data/memory/snapshots.json"

def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

@dataclass
class MemoryStore:
    events_path: str = DEFAULT_EVENTS_PATH
    snapshots_path: str = DEFAULT_SNAPSHOTS_PATH

    def append_event(self, event: Dict[str, Any]) -> None:
        _ensure_dir(self.events_path)
        event = {**event, "logged_at": datetime.utcnow().isoformat()}
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

    def load_events(self, company_id: Optional[str] = None, limit: int = 5000) -> List[Dict[str, Any]]:
        if not os.path.exists(self.events_path):
            return []
        events: List[Dict[str, Any]] = []
        with open(self.events_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                e = json.loads(line)
                if company_id is None or e.get("company_id") == company_id:
                    events.append(e)
        return events[-limit:]

    def save_snapshot(self, company_id: str, snapshot: Dict[str, Any]) -> None:
        _ensure_dir(self.snapshots_path)
        data = {}
        if os.path.exists(self.snapshots_path):
            with open(self.snapshots_path, "r", encoding="utf-8") as f:
                raw = f.read().strip()
                data = json.loads(raw) if raw else {}
        data[company_id] = snapshot
        with open(self.snapshots_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_snapshot(self, company_id: str) -> Dict[str, Any]:
        if not os.path.exists(self.snapshots_path):
            return {}
        with open(self.snapshots_path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
            data = json.loads(raw) if raw else {}
        return data.get(company_id, {})

class ReflectionEngine:
    """
    Logs:
      - disruptions
      - decisions
      - mitigation executions
    Evaluates:
      - success rates by action type
      - ROI vs predicted ROI
      - what patterns lead to failures
    Improves:
      - provides "insights" used by planning to bias future recommendations
    """
    def __init__(self, store: Optional[MemoryStore] = None):
        self.store = store or MemoryStore()

    async def log_disruption(self, company_id: str, disruption: Dict[str, Any]) -> None:
        self.store.append_event({
            "event_type": "disruption",
            "company_id": company_id,
            "disruption": disruption
        })

    async def log_decision_record(self, company_id: str, decision: Dict[str, Any]) -> None:
        self.store.append_event({
            "event_type": "decision",
            "company_id": company_id,
            "decision": decision
        })

    async def log_mitigation_execution(self, company_id: str, execution: Dict[str, Any]) -> None:
        self.store.append_event({
            "event_type": "execution",
            "company_id": company_id,
            "execution": execution
        })

    async def evaluate_mitigation_success(self, company_id: str) -> Dict[str, Any]:
        events = self.store.load_events(company_id=company_id)

        executions = [e for e in events if e.get("event_type") == "execution"]
        if not executions:
            return {"company_id": company_id, "summary": "no executions logged yet"}

        # Aggregate success/failure by action_type
        stats: Dict[str, Dict[str, Any]] = {}
        for e in executions:
            ex = e.get("execution", {})
            a_type = ex.get("action_type", "unknown")
            status = ex.get("status", "unknown")
            predicted_roi = ex.get("predicted_roi", None)
            realized_roi = ex.get("realized_roi", None)

            if a_type not in stats:
                stats[a_type] = {"total": 0, "completed": 0, "failed": 0, "roi_pairs": []}

            stats[a_type]["total"] += 1
            if status == "completed":
                stats[a_type]["completed"] += 1
            elif status == "failed":
                stats[a_type]["failed"] += 1

            if predicted_roi is not None and realized_roi is not None:
                stats[a_type]["roi_pairs"].append((predicted_roi, realized_roi))

        # compute summary and save snapshot
        summary = {}
        for a_type, d in stats.items():
            success_rate = d["completed"] / d["total"] if d["total"] else 0
            roi_bias = 0.0
            if d["roi_pairs"]:
                diffs = [real - pred for pred, real in d["roi_pairs"]]
                roi_bias = sum(diffs) / len(diffs)
            summary[a_type] = {
                "success_rate": round(success_rate, 3),
                "roi_bias": round(roi_bias, 3),
                "total": d["total"]
            }

        snapshot = {
            "company_id": company_id,
            "evaluated_at": datetime.utcnow().isoformat(),
            "by_action_type": summary
        }
        self.store.save_snapshot(company_id, snapshot)
        return snapshot

    async def get_company_insights(self, company_id: str) -> Dict[str, Any]:
        """
        Used by planning layer to adjust future decisions.
        Example outputs:
          - action_type_penalties: {"rerouting": 0.05}  (if rerouting fails often)
          - action_type_boosts: {"buffering": 0.08}    (if buffering succeeds often)
        """
        snap = self.store.load_snapshot(company_id)
        if not snap:
            return {
                "company_id": company_id,
                "note": "no snapshot yet",
                "action_type_boosts": {},
                "action_type_penalties": {}
            }

        boosts = {}
        penalties = {}
        for a_type, metrics in snap.get("by_action_type", {}).items():
            sr = metrics.get("success_rate", 0.0)
            roi_bias = metrics.get("roi_bias", 0.0)

            # Simple policy:
            # - if success rate high, boost that action type
            # - if success rate low, penalize that action type
            # - if ROI is overestimated (negative bias), penalize a bit
            if sr >= 0.8:
                boosts[a_type] = 0.08
            if sr <= 0.5 and metrics.get("total", 0) >= 2:
                penalties[a_type] = 0.08
            if roi_bias < -0.2:
                penalties[a_type] = max(penalties.get(a_type, 0), 0.05)

        return {
            "company_id": company_id,
            "evaluated_at": snap.get("evaluated_at"),
            "action_type_boosts": boosts,
            "action_type_penalties": penalties
        }