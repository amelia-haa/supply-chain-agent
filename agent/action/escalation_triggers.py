# src/action/escalation_engine.py
import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import logging

from agent.models import Company, Supplier, RiskAssessment, Disruption, MitigationAction
from agent.action.email_generator import EmailGenerator
from agent.action.workflow_manager import WorkflowManager

logger = logging.getLogger(__name__)


class EscalationLevel(Enum):
    NONE = "none"
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class TriggerType(Enum):
    RISK_SCORE_SPIKE = "risk_score_spike"
    SUPPLIER_PERFORMANCE_DROP = "supplier_performance_drop"
    STOCKOUT_RISK = "stockout_risk"
    LEAD_TIME_SPIKE = "lead_time_spike"
    DISRUPTION_SEVERE = "disruption_severe"
    REVENUE_AT_RISK_BREACH = "revenue_at_risk_breach"
    PO_DUE_SOON_HIGH_RISK = "po_due_soon_high_risk"


@dataclass
class EscalationTriggerEvent:
    trigger_type: str
    level: str
    message: str
    evidence: Dict[str, Any]
    created_at: datetime


class EscalationEngine:
    """
    Central escalation trigger engine:
    - Evaluate signals
    - Produce trigger events
    - Execute escalation actions (emails + workflows)
    """

    def __init__(self):
        self.email_generator = EmailGenerator()
        self.workflow_manager = WorkflowManager()

        # Default trigger thresholds (override per-company if you want)
        self.thresholds = {
            # risk
            "risk_spike_delta": 0.12,         # +12% increase vs previous
            "critical_risk_score": 0.80,
            "warning_risk_score": 0.65,

            # supplier performance
            "otif_drop_critical": 0.15,       # 15% drop
            "otif_drop_warning": 0.08,

            # lead time
            "lead_time_spike_warning": 0.20,  # +20%
            "lead_time_spike_critical": 0.35, # +35%

            # stockout
            "stockout_days_critical": 7,
            "stockout_days_warning": 14,

            # revenue at risk
            "revenue_at_risk_warning": 250_000,
            "revenue_at_risk_critical": 1_000_000,

            # PO timing
            "po_due_soon_days": 10
        }

    async def evaluate(
        self,
        company: Company,
        suppliers: List[Supplier],
        risk_assessment: RiskAssessment,
        signals: Dict[str, Any],
        disruption: Optional[Disruption] = None
    ) -> Dict[str, Any]:
        """
        signals is a flexible dict you pass from your monitoring/planning layer.
        Examples keys:
          - previous_risk_score
          - supplier_otif: {supplier_id: {"current":0.91, "baseline":0.98}}
          - stockout_projection_days: {sku_or_component: 9}
          - lead_time_change_pct: {supplier_id: 0.28}
          - revenue_at_risk: 1200000
          - po_due_days: {po_id: 5}
          - po_supplier_risk: {po_id: 0.82}
        """
        events: List[EscalationTriggerEvent] = []

        # 1) risk score spike / high
        events += self._check_risk(company, risk_assessment, signals)

        # 2) supplier performance drops
        events += self._check_supplier_performance(signals)

        # 3) stockout risk
        events += self._check_stockouts(signals)

        # 4) lead time spikes
        events += self._check_lead_time(signals)

        # 5) disruption severity
        if disruption is not None:
            events += self._check_disruption(disruption)

        # 6) revenue-at-risk breach
        events += self._check_revenue_at_risk(signals)

        # 7) PO due soon + high supplier risk
        events += self._check_po_due_soon(signals)

        # Deduplicate + choose overall level
        overall = self._overall_escalation_level(events)

        return {
            "company_id": company.id,
            "evaluated_at": datetime.utcnow(),
            "overall_level": overall.value,
            "events": [e.__dict__ for e in events],
        }

    async def execute_escalations(
        self,
        company: Company,
        evaluation: Dict[str, Any],
        action_context: Optional[MitigationAction] = None
    ) -> Dict[str, Any]:
        """
        Execute the escalation actions based on evaluation:
        - CRITICAL: escalation email + escalation workflow
        - WARNING: workflow + summary email
        - INFO: summary email only
        """
        overall = EscalationLevel(evaluation["overall_level"])
        events = evaluation.get("events", [])

        if overall == EscalationLevel.NONE:
            return {"executed": False, "reason": "No escalation triggers", "evaluation": evaluation}

        # Create a lightweight action object if not provided
        if action_context is None:
            action_context = MitigationAction(
                company_id=company.id,
                risk_assessment_id=1,
                action_type="escalation",
                title=f"Escalation Triggered ({overall.value.upper()})",
                description="Auto-generated escalation based on trigger evaluation",
                priority_level="critical" if overall == EscalationLevel.CRITICAL else "high",
                estimated_cost=0,
                estimated_benefit=0,
                roi_score=0,
                implementation_time_days=0,
                urgency_score=0.9 if overall == EscalationLevel.CRITICAL else 0.6
            )

        results: Dict[str, Any] = {"executed": True, "overall_level": overall.value, "actions": []}

        # Always: send escalation notifications (you already have generator)
        email_payload = {
            "overall_level": overall.value,
            "trigger_count": len(events),
            "events": events
        }

        # CRITICAL -> use escalation email template
        if overall == EscalationLevel.CRITICAL:
            emails = await self.email_generator.generate_escalation_notifications(company, action_context)
            results["actions"].append({"type": "escalation_emails", "result": emails})

            wf = await self.workflow_manager.create_escalation_workflow(company, action_context)
            results["actions"].append({"type": "escalation_workflow", "result": wf})

        # WARNING -> summary email + workflow
        elif overall == EscalationLevel.WARNING:
            summary_email = await self._send_summary_escalation(company, action_context, email_payload, priority="high")
            results["actions"].append({"type": "summary_email", "result": summary_email})

            wf = await self.workflow_manager.create_escalation_workflow(company, action_context)
            results["actions"].append({"type": "escalation_workflow", "result": wf})

        # INFO -> summary only
        else:
            summary_email = await self._send_summary_escalation(company, action_context, email_payload, priority="medium")
            results["actions"].append({"type": "summary_email", "result": summary_email})

        return results

    # ----------------------------
    # Trigger checks (sync helpers)
    # ----------------------------

    def _check_risk(self, company: Company, risk_assessment: RiskAssessment, signals: Dict[str, Any]) -> List[EscalationTriggerEvent]:
        events = []
        current = getattr(risk_assessment, "overall_risk_score", None)
        if current is None:
            current = signals.get("current_risk_score")

        prev = signals.get("previous_risk_score")
        if current is None:
            return events

        # spike
        if prev is not None and (current - prev) >= self.thresholds["risk_spike_delta"]:
            events.append(EscalationTriggerEvent(
                trigger_type=TriggerType.RISK_SCORE_SPIKE.value,
                level=EscalationLevel.WARNING.value if current < self.thresholds["critical_risk_score"] else EscalationLevel.CRITICAL.value,
                message=f"Risk score spiked from {prev:.2f} to {current:.2f}.",
                evidence={"previous": prev, "current": current, "delta": current - prev},
                created_at=datetime.utcnow()
            ))

        # absolute
        if current >= self.thresholds["critical_risk_score"]:
            events.append(EscalationTriggerEvent(
                trigger_type="risk_score_critical",
                level=EscalationLevel.CRITICAL.value,
                message=f"Risk score is CRITICAL at {current:.2f}.",
                evidence={"current": current, "threshold": self.thresholds["critical_risk_score"]},
                created_at=datetime.utcnow()
            ))
        elif current >= self.thresholds["warning_risk_score"]:
            events.append(EscalationTriggerEvent(
                trigger_type="risk_score_warning",
                level=EscalationLevel.WARNING.value,
                message=f"Risk score is elevated at {current:.2f}.",
                evidence={"current": current, "threshold": self.thresholds["warning_risk_score"]},
                created_at=datetime.utcnow()
            ))

        return events

    def _check_supplier_performance(self, signals: Dict[str, Any]) -> List[EscalationTriggerEvent]:
        events = []
        otif = signals.get("supplier_otif", {})
        for supplier_id, data in otif.items():
            cur = data.get("current")
            base = data.get("baseline")
            if cur is None or base is None:
                continue
            drop = base - cur
            if drop >= self.thresholds["otif_drop_critical"]:
                events.append(EscalationTriggerEvent(
                    trigger_type=TriggerType.SUPPLIER_PERFORMANCE_DROP.value,
                    level=EscalationLevel.CRITICAL.value,
                    message=f"Supplier {supplier_id} OTIF dropped by {drop:.2%} (baseline {base:.2%} → current {cur:.2%}).",
                    evidence={"supplier_id": supplier_id, "baseline": base, "current": cur, "drop": drop},
                    created_at=datetime.utcnow()
                ))
            elif drop >= self.thresholds["otif_drop_warning"]:
                events.append(EscalationTriggerEvent(
                    trigger_type=TriggerType.SUPPLIER_PERFORMANCE_DROP.value,
                    level=EscalationLevel.WARNING.value,
                    message=f"Supplier {supplier_id} OTIF drop warning: {drop:.2%}.",
                    evidence={"supplier_id": supplier_id, "baseline": base, "current": cur, "drop": drop},
                    created_at=datetime.utcnow()
                ))
        return events

    def _check_stockouts(self, signals: Dict[str, Any]) -> List[EscalationTriggerEvent]:
        events = []
        proj = signals.get("stockout_projection_days", {})
        for item, days in proj.items():
            if days is None:
                continue
            if days <= self.thresholds["stockout_days_critical"]:
                events.append(EscalationTriggerEvent(
                    trigger_type=TriggerType.STOCKOUT_RISK.value,
                    level=EscalationLevel.CRITICAL.value,
                    message=f"Stockout risk CRITICAL for {item}: projected stockout in {days} days.",
                    evidence={"item": item, "days": days},
                    created_at=datetime.utcnow()
                ))
            elif days <= self.thresholds["stockout_days_warning"]:
                events.append(EscalationTriggerEvent(
                    trigger_type=TriggerType.STOCKOUT_RISK.value,
                    level=EscalationLevel.WARNING.value,
                    message=f"Stockout risk warning for {item}: projected stockout in {days} days.",
                    evidence={"item": item, "days": days},
                    created_at=datetime.utcnow()
                ))
        return events

    def _check_lead_time(self, signals: Dict[str, Any]) -> List[EscalationTriggerEvent]:
        events = []
        lt = signals.get("lead_time_change_pct", {})
        for supplier_id, pct in lt.items():
            if pct is None:
                continue
            if pct >= self.thresholds["lead_time_spike_critical"]:
                events.append(EscalationTriggerEvent(
                    trigger_type=TriggerType.LEAD_TIME_SPIKE.value,
                    level=EscalationLevel.CRITICAL.value,
                    message=f"Lead time spike CRITICAL for supplier {supplier_id}: +{pct:.0%}.",
                    evidence={"supplier_id": supplier_id, "lead_time_change_pct": pct},
                    created_at=datetime.utcnow()
                ))
            elif pct >= self.thresholds["lead_time_spike_warning"]:
                events.append(EscalationTriggerEvent(
                    trigger_type=TriggerType.LEAD_TIME_SPIKE.value,
                    level=EscalationLevel.WARNING.value,
                    message=f"Lead time spike warning for supplier {supplier_id}: +{pct:.0%}.",
                    evidence={"supplier_id": supplier_id, "lead_time_change_pct": pct},
                    created_at=datetime.utcnow()
                ))
        return events

    def _check_disruption(self, disruption: Disruption) -> List[EscalationTriggerEvent]:
        if disruption.severity_score >= 0.85:
            return [EscalationTriggerEvent(
                trigger_type=TriggerType.DISRUPTION_SEVERE.value,
                level=EscalationLevel.CRITICAL.value,
                message=f"Severe disruption detected: {disruption.title} (severity {disruption.severity_score:.2f}).",
                evidence={"title": disruption.title, "type": disruption.disruption_type, "severity": disruption.severity_score},
                created_at=datetime.utcnow()
            )]
        if disruption.severity_score >= 0.70:
            return [EscalationTriggerEvent(
                trigger_type=TriggerType.DISRUPTION_SEVERE.value,
                level=EscalationLevel.WARNING.value,
                message=f"Disruption warning: {disruption.title} (severity {disruption.severity_score:.2f}).",
                evidence={"title": disruption.title, "type": disruption.disruption_type, "severity": disruption.severity_score},
                created_at=datetime.utcnow()
            )]
        return []

    def _check_revenue_at_risk(self, signals: Dict[str, Any]) -> List[EscalationTriggerEvent]:
        events = []
        rar = signals.get("revenue_at_risk")
        if rar is None:
            return events
        if rar >= self.thresholds["revenue_at_risk_critical"]:
            events.append(EscalationTriggerEvent(
                trigger_type=TriggerType.REVENUE_AT_RISK_BREACH.value,
                level=EscalationLevel.CRITICAL.value,
                message=f"Revenue-at-risk CRITICAL: ${rar:,.0f}.",
                evidence={"revenue_at_risk": rar, "threshold": self.thresholds["revenue_at_risk_critical"]},
                created_at=datetime.utcnow()
            ))
        elif rar >= self.thresholds["revenue_at_risk_warning"]:
            events.append(EscalationTriggerEvent(
                trigger_type=TriggerType.REVENUE_AT_RISK_BREACH.value,
                level=EscalationLevel.WARNING.value,
                message=f"Revenue-at-risk warning: ${rar:,.0f}.",
                evidence={"revenue_at_risk": rar, "threshold": self.thresholds["revenue_at_risk_warning"]},
                created_at=datetime.utcnow()
            ))
        return events

    def _check_po_due_soon(self, signals: Dict[str, Any]) -> List[EscalationTriggerEvent]:
        events = []
        po_due = signals.get("po_due_days", {})         # {po_id: days}
        po_risk = signals.get("po_supplier_risk", {})   # {po_id: risk_score}
        for po_id, days in po_due.items():
            if days is None:
                continue
            if days <= self.thresholds["po_due_soon_days"]:
                r = po_risk.get(po_id, 0.0)
                if r >= 0.8:
                    events.append(EscalationTriggerEvent(
                        trigger_type=TriggerType.PO_DUE_SOON_HIGH_RISK.value,
                        level=EscalationLevel.CRITICAL.value,
                        message=f"PO {po_id} due in {days} days with high supplier risk ({r:.2f}).",
                        evidence={"po_id": po_id, "days": days, "supplier_risk": r},
                        created_at=datetime.utcnow()
                    ))
                elif r >= 0.65:
                    events.append(EscalationTriggerEvent(
                        trigger_type=TriggerType.PO_DUE_SOON_HIGH_RISK.value,
                        level=EscalationLevel.WARNING.value,
                        message=f"PO {po_id} due soon ({days} days) with elevated supplier risk ({r:.2f}).",
                        evidence={"po_id": po_id, "days": days, "supplier_risk": r},
                        created_at=datetime.utcnow()
                    ))
        return events

    def _overall_escalation_level(self, events: List[EscalationTriggerEvent]) -> EscalationLevel:
        if any(e.level == EscalationLevel.CRITICAL.value for e in events):
            return EscalationLevel.CRITICAL
        if any(e.level == EscalationLevel.WARNING.value for e in events):
            return EscalationLevel.WARNING
        if any(e.level == EscalationLevel.INFO.value for e in events):
            return EscalationLevel.INFO
        return EscalationLevel.NONE

    async def _send_summary_escalation(self, company: Company, action: MitigationAction, payload: Dict[str, Any], priority: str) -> Dict[str, Any]:
        """
        Uses your existing escalation email generator, but you can also
        create a dedicated "summary escalation" email template later.
        """
        # For now, reuse escalation notifications (or build a separate summary template)
        return await self.email_generator.generate_escalation_notifications(company, action)