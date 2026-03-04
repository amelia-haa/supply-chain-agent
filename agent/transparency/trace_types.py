# src/transparency/trace_types.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

@dataclass
class ConstraintViolation:
    code: str
    severity: str  # "low" | "medium" | "high" | "critical"
    message: str
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ReasonTrace:
    """
    A structured trace for explaining why an agent made a decision.
    """
    trace_id: str
    created_at: str
    decision_stage: str  # e.g. "planning" | "action_layer" | "po_adjustment"
    decision_name: str   # human-readable name

    inputs_summary: Dict[str, Any] = field(default_factory=dict)
    reasoning_steps: List[Dict[str, Any]] = field(default_factory=list)
    risk_justification: Dict[str, Any] = field(default_factory=dict)

    constraints_checked: List[str] = field(default_factory=list)
    constraint_violations: List[ConstraintViolation] = field(default_factory=list)

    bias_checks: List[str] = field(default_factory=list)
    bias_findings: List[Dict[str, Any]] = field(default_factory=list)

    override_policy: Dict[str, Any] = field(default_factory=dict)
    override_required: bool = False
    override_reason: Optional[str] = None

    final_decision: Dict[str, Any] = field(default_factory=dict)

    def add_step(self, title: str, details: Dict[str, Any]) -> None:
        self.reasoning_steps.append({
            "title": title,
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        })

    def add_violation(self, v: ConstraintViolation) -> None:
        self.constraint_violations.append(v)