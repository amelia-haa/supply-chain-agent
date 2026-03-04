"""
src/action/action_executor.py

Execution Engine / Autonomous Action Layer Orchestrator

Responsibilities:
- Decide execution mode (automatic / human approval / executive approval)
- Execute mitigation actions: resourcing, buffering, rerouting, negotiation, escalation
- Generate comprehensive Purchase Order (PO) adjustment suggestions
- Apply PO adjustments when allowed
- Trigger escalations using rules + optional LLM reasoning
- Create workflows and send notifications via EmailGenerator
- Track audit trail + execution logs

This works with your:
- src/action/email_generator.py
- src/action/workflow_manager.py
- src/action/erp_integrator.py  (the comprehensive one)
"""

from __future__ import annotations
from agent.transparency import TransparencyEngine
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import google.generativeai as genai

from agent.config import settings
from agent.models import Company, Supplier, Disruption, RiskAssessment, MitigationAction

from agent.action.email_generator import EmailGenerator
from agent.action.workflow_manager import WorkflowManager
from agent.action.erp_integrator import ERPIntegrator

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    AUTOMATIC = "automatic"
    HUMAN_APPROVAL = "human_approval"
    EXECUTIVE_APPROVAL = "executive_approval"


class ExecutionStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING_APPROVAL = "waiting_approval"
    CANCELLED = "cancelled"


@dataclass
class ExecutionConfig:
    automatic_execution_threshold: float = 0.80
    human_approval_threshold: float = 0.50
    executive_approval_threshold: float = 0.30

    # Escalation rules
    escalation_risk_threshold: float = 0.80
    escalation_severe_disruption_threshold: float = 0.75
    escalation_po_cost_impact_threshold: float = 250_000.0   # if PO adjustments are expensive
    escalation_service_risk_threshold: float = 0.70          # risk of stockout/service failure

    # PO policy
    max_auto_po_total_cost_increase: float = 100_000.0       # beyond this → approval
    allow_auto_apply_po_changes: bool = True


class ActionExecutor:
    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()

        # LLM optional
        self.llm_enabled = True
        try:
            genai.configure(api_key=settings.google_api_key)
            self.model = genai.GenerativeModel(settings.gemini_model)
        except Exception as e:
            self.llm_enabled = False
            self.model = None
            logger.warning(f"Gemini init failed; continuing without LLM. Error: {e}")

        # Components
        self.email_generator = EmailGenerator()
        self.workflow_manager = WorkflowManager()
        self.erp = ERPIntegrator()

        # execution tracking
        self.active_executions: Dict[str, Dict[str, Any]] = {}
        self.transparency = TransparencyEngine()

    # ---------------------------
    # Public API
    # ---------------------------

    async def execute_mitigation_actions(
        self,
        company: Company,
        actions: List[MitigationAction],
        risk_assessment: RiskAssessment,
        suppliers: Optional[List[Supplier]] = None,
        disruptions: Optional[List[Disruption]] = None,
        auto_execute: bool = False
    ) -> Dict[str, Any]:
        """Execute a list of mitigation actions with gating + logging."""
        suppliers = suppliers or []
        disruptions = disruptions or []

        results: List[Dict[str, Any]] = []
        for action in actions:
            mode = await self._determine_execution_mode(company, action, risk_assessment, auto_execute)
            res = await self._execute_action(company, action, mode, risk_assessment, suppliers, disruptions)
            results.append(res)

        summary = self._summarize(results)

        # Always generate follow-ups for non-auto approvals
        follow_up = await self._create_follow_up_tasks(company, results)

        return {
            "company_id": str(company.id),
            "execution_timestamp": datetime.utcnow(),
            "results": results,
            "summary": summary,
            "follow_up_tasks": follow_up,
            "next_review_date": datetime.utcnow() + timedelta(days=7),
        }

    async def generate_and_execute_po_adjustments(
        self,
        company: Company,
        risk_assessment: RiskAssessment,
        suppliers: Optional[List[Supplier]] = None,
        disruptions: Optional[List[Disruption]] = None,
        auto_execute: bool = False,
        horizon_days: int = 90,
    ) -> Dict[str, Any]:
        """
        Comprehensive PO adjustment engine:
        1) Generate PO adjustment suggestions
        2) Decide escalation/approval path
        3) Create workflow tasks + emails
        4) Optionally apply ERP changes (if allowed)
        """
        suppliers = suppliers or []
        disruptions = disruptions or []

        suggestions = await self.erp.generate_po_adjustment_suggestions(
            company=company,
            risk_assessment=risk_assessment,
            disruptions=disruptions,
            suppliers=suppliers,
            horizon_days=horizon_days,
        )

        # Decide if we must escalate or require approval
        gate = await self._gate_po_adjustments(company, risk_assessment, disruptions, suggestions, auto_execute)

        # Create workflows regardless (so humans can review)
        workflow = await self.workflow_manager.create_po_adjustment_workflow(company, suggestions, gate)

        # Notify procurement + ops
        po_email_pack = await self.email_generator.generate_inventory_notifications(
            company,
            inventory_recommendations={
                "type": "PO Adjustment Suggestions",
                "gate": gate,
                "portfolio_summary": suggestions.get("portfolio_summary"),
            },
        )

        applied_result = None
        if gate["decision"] == "apply_automatically" and self.config.allow_auto_apply_po_changes:
            # Convert suggestions → concrete ERP "adjustments"
            adjustments = self._compile_po_adjustments_payload(suggestions)
            applied_result = await self.erp.apply_po_adjustments(company, adjustments)

        # Escalation triggers (exec attention)
        escalation = await self._maybe_trigger_escalation(company, risk_assessment, disruptions, suggestions, gate)

        return {
            "company_id": str(company.id),
            "timestamp": datetime.utcnow(),
            "po_suggestions": suggestions,
            "gate": gate,
            "workflow": workflow,
            "notification_emails": po_email_pack,
            "applied_result": applied_result,
            "escalation": escalation,
        }

    # ---------------------------
    # Core Execution
    # ---------------------------

    async def _execute_action(
        self,
        company: Company,
        action: MitigationAction,
        mode: ExecutionMode,
        risk_assessment: RiskAssessment,
        suppliers: List[Supplier],
        disruptions: List[Disruption],
    ) -> Dict[str, Any]:
        execution_id = f"{getattr(action, 'id', 'NA')}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        self.active_executions[execution_id] = {
            "action_id": getattr(action, "id", None),
            "action_type": getattr(action, "action_type", None),
            "mode": mode.value,
            "status": ExecutionStatus.IN_PROGRESS.value,
            "started_at": datetime.utcnow(),
        }

        try:
            # approval gating
            if mode != ExecutionMode.AUTOMATIC:
                self.active_executions[execution_id]["status"] = ExecutionStatus.WAITING_APPROVAL.value
                # create workflow for approval and stop here (don’t execute)
                approval_workflow = await self.workflow_manager.create_approval_workflow(company, action, mode.value)
                return {
                    "execution_id": execution_id,
                    "status": ExecutionStatus.WAITING_APPROVAL.value,
                    "mode": mode.value,
                    "action_id": getattr(action, "id", None),
                    "action_type": getattr(action, "action_type", None),
                    "approval_workflow": approval_workflow,
                    "message": "Action requires approval before execution.",
                }

            # automatic execution routing
            atype = getattr(action, "action_type", "")
            if atype in ("resourcing", "negotiation"):
                result = await self._execute_supplier_actions(company, action)
            elif atype == "buffering":
                result = await self._execute_inventory_actions(company, action)
            elif atype == "rerouting":
                result = await self._execute_logistics_actions(company, action)
            elif atype == "escalation":
                result = await self._execute_escalation_actions(company, action)
            else:
                result = await self._execute_general_actions(company, action)

            self.active_executions[execution_id]["status"] = ExecutionStatus.COMPLETED.value
            self.active_executions[execution_id]["completed_at"] = datetime.utcnow()

            return {
                "execution_id": execution_id,
                "status": ExecutionStatus.COMPLETED.value,
                "mode": mode.value,
                "action_id": getattr(action, "id", None),
                "action_type": atype,
                "result": result,
            }

        except Exception as e:
            logger.exception("Execution failed")
            self.active_executions[execution_id]["status"] = ExecutionStatus.FAILED.value
            return {
                "execution_id": execution_id,
                "status": ExecutionStatus.FAILED.value,
                "mode": mode.value,
                "action_id": getattr(action, "id", None),
                "action_type": getattr(action, "action_type", None),
                "error": str(e),
            }

    # ---------------------------
    # Execution subflows
    # ---------------------------

    async def _execute_supplier_actions(self, company: Company, action: MitigationAction) -> Dict[str, Any]:
        if action.action_type == "resourcing":
            emails = await self.email_generator.generate_supplier_outreach_emails(company, action)
            wf = await self.workflow_manager.create_supplier_qualification_workflow(company, action)
            erp = await self.erp.update_supplier_requirements(company, action)
            return {"emails": emails, "workflow": wf, "erp": erp}

        if action.action_type == "negotiation":
            emails = await self.email_generator.generate_negotiation_emails(company, action)
            wf = await self.workflow_manager.create_negotiation_workflow(company, action)
            return {"emails": emails, "workflow": wf}

        # fallback
        emails = await self.email_generator.generate_supplier_communications(company, action)
        wf = await self.workflow_manager.create_general_workflow(company, action)
        return {"emails": emails, "workflow": wf}

    async def _execute_inventory_actions(self, company: Company, action: MitigationAction) -> Dict[str, Any]:
        recs = await self.erp.calculate_optimal_inventory(company, action)
        erp_update = await self.erp.update_inventory_levels(company, recs)
        emails = await self.email_generator.generate_inventory_notifications(company, recs)
        wf = await self.workflow_manager.create_inventory_change_workflow(company, action, recs)
        return {"recommendations": recs, "erp_update": erp_update, "emails": emails, "workflow": wf}

    async def _execute_logistics_actions(self, company: Company, action: MitigationAction) -> Dict[str, Any]:
        routes = await self.erp.identify_alternative_routes(company, action)
        erp_update = await self.erp.update_logistics_routes(company, routes)
        emails = await self.email_generator.generate_logistics_notifications(company, routes)
        wf = await self.workflow_manager.create_logistics_reroute_workflow(company, action, routes)
        return {"routes": routes, "erp_update": erp_update, "emails": emails, "workflow": wf}

    async def _execute_escalation_actions(self, company: Company, action: MitigationAction) -> Dict[str, Any]:
        emails = await self.email_generator.generate_escalation_notifications(company, action)
        wf = await self.workflow_manager.create_escalation_workflow(company, action)
        return {"emails": emails, "workflow": wf, "logged": True}

    async def _execute_general_actions(self, company: Company, action: MitigationAction) -> Dict[str, Any]:
        emails = await self.email_generator.generate_general_communications(company, action)
        wf = await self.workflow_manager.create_general_workflow(company, action)
        return {"emails": emails, "workflow": wf}

    # ---------------------------
    # Decisioning / Gating
    # ---------------------------

    async def _determine_execution_mode(
        self,
        company: Company,
        action: MitigationAction,
        risk_assessment: RiskAssessment,
        auto_execute: bool
    ) -> ExecutionMode:
        conf = await self._calculate_execution_confidence(company, action, risk_assessment)

        if auto_execute and conf >= self.config.automatic_execution_threshold:
            return ExecutionMode.AUTOMATIC
        if conf >= self.config.human_approval_threshold:
            return ExecutionMode.HUMAN_APPROVAL
        return ExecutionMode.EXECUTIVE_APPROVAL

    async def _calculate_execution_confidence(
        self,
        company: Company,
        action: MitigationAction,
        risk_assessment: RiskAssessment
    ) -> float:
        base = 0.5
        by_type = {
            "buffering": 0.90,
            "negotiation": 0.60,
            "resourcing": 0.70,
            "rerouting": 0.80,
            "escalation": 0.40,
        }
        type_conf = by_type.get(getattr(action, "action_type", ""), 0.55)

        priority = getattr(action, "priority_level", "medium")
        priority_mult = {"critical": 1.2, "high": 1.1, "medium": 1.0, "low": 0.9}.get(priority, 1.0)

        risk = getattr(risk_assessment, "overall_risk_score", None)
        if risk is None:
            risk = getattr(risk_assessment, "composite_risk_score", 0.6)

        appetite = getattr(company, "risk_appetite", 0.5)

        score = (base + type_conf) * priority_mult + min(0.2, risk * 0.2) + appetite * 0.1
        return max(0.0, min(1.0, score))

    async def _gate_po_adjustments(
        self,
        company: Company,
        risk_assessment: RiskAssessment,
        disruptions: List[Disruption],
        suggestions: Dict[str, Any],
        auto_execute: bool
    ) -> Dict[str, Any]:
        """
        Decide whether PO changes can be applied automatically or require approval.
        """
        portfolio = suggestions.get("portfolio_summary", {}) or {}
        cost = float(portfolio.get("estimated_incremental_cost", 0.0))
        risk = getattr(risk_assessment, "overall_risk_score", None)
        if risk is None:
            risk = getattr(risk_assessment, "composite_risk_score", 0.6)

        severe = any(getattr(d, "severity_score", 0) >= self.config.escalation_severe_disruption_threshold for d in disruptions)

        # Decision rules (simple but practical)
        if not auto_execute:
            return {
                "decision": "require_human_review",
                "reason": "auto_execute=False",
                "estimated_incremental_cost": cost,
                "risk_score": risk,
                "severe_disruption": severe,
            }

        if cost > self.config.max_auto_po_total_cost_increase:
            return {
                "decision": "require_human_review",
                "reason": f"Estimated cost increase (${cost:,.0f}) exceeds auto limit.",
                "estimated_incremental_cost": cost,
                "risk_score": risk,
                "severe_disruption": severe,
            }

        # If extreme risk, we may still want humans even if cost is low
        if risk >= self.config.escalation_risk_threshold or severe:
            return {
                "decision": "require_executive_review",
                "reason": "High systemic risk / severe disruption; require executive oversight.",
                "estimated_incremental_cost": cost,
                "risk_score": risk,
                "severe_disruption": severe,
            }

        return {
            "decision": "apply_automatically",
            "reason": "Within policy thresholds for automatic application.",
            "estimated_incremental_cost": cost,
            "risk_score": risk,
            "severe_disruption": severe,
        }

    async def _maybe_trigger_escalation(
        self,
        company: Company,
        risk_assessment: RiskAssessment,
        disruptions: List[Disruption],
        suggestions: Dict[str, Any],
        gate: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Escalate if: high risk / severe disruption / expensive PO changes / high service risk.
        """
        portfolio = suggestions.get("portfolio_summary", {}) or {}
        cost = float(portfolio.get("estimated_incremental_cost", 0.0))

        risk = getattr(risk_assessment, "overall_risk_score", None)
        if risk is None:
            risk = getattr(risk_assessment, "composite_risk_score", 0.6)

        severe = any(getattr(d, "severity_score", 0) >= self.config.escalation_severe_disruption_threshold for d in disruptions)

        must_escalate = (
            risk >= self.config.escalation_risk_threshold
            or severe
            or cost >= self.config.escalation_po_cost_impact_threshold
            or gate.get("decision") == "require_executive_review"
        )

        if not must_escalate:
            return None

        # Build a synthetic “escalation action” so your email/workflow code can reuse it
        escalation_action = MitigationAction(
            company_id=company.id,
            risk_assessment_id=getattr(risk_assessment, "id", 1),
            action_type="escalation",
            title="Escalation: Procurement & Service Risk Requires Attention",
            description=f"Escalation triggered. Risk={risk:.2f}, Severe={severe}, Estimated PO cost impact=${cost:,.0f}. Gate={gate.get('decision')}.",
            priority_level="critical" if severe or risk > 0.9 else "high",
            estimated_cost=0,
            estimated_benefit=0,
            roi_score=0,
            implementation_time_days=1,
            urgency_score=0.9 if severe else 0.7,
        )

        # Optional LLM justification for exec summary
        llm_summary = None
        if self.llm_enabled and self.model is not None:
            llm_summary = await self._llm_escalation_summary(company, risk_assessment, disruptions, portfolio, gate)

        emails = await self.email_generator.generate_escalation_notifications(company, escalation_action)
        wf = await self.workflow_manager.create_escalation_workflow(company, escalation_action)

        return {
            "triggered": True,
            "reason": escalation_action.description,
            "llm_summary": llm_summary,
            "emails": emails,
            "workflow": wf,
        }

    async def _llm_escalation_summary(
        self,
        company: Company,
        risk_assessment: RiskAssessment,
        disruptions: List[Disruption],
        portfolio: Dict[str, Any],
        gate: Dict[str, Any],
    ) -> Optional[str]:
        if not self.llm_enabled or self.model is None:
            return None

        risk = getattr(risk_assessment, "overall_risk_score", None)
        if risk is None:
            risk = getattr(risk_assessment, "composite_risk_score", 0.6)

        top = []
        for d in disruptions[:5]:
            top.append({
                "type": getattr(d, "disruption_type", "unknown"),
                "severity": getattr(d, "severity_score", None),
                "regions": getattr(d, "affected_regions", None),
            })

        prompt = f"""
Write an executive escalation note (6-8 sentences).
Company: {company.name}
Risk score: {risk}
Top disruptions: {top}
PO portfolio impact: {portfolio}
Decision gate: {gate}

Be direct:
- what is happening
- why it matters (service + cost)
- what decision is needed now
"""
        try:
            resp = await self.model.generate_content_async(prompt)
            return resp.text.strip()
        except Exception as e:
            logger.warning(f"LLM escalation summary failed: {e}")
            return None

    # ---------------------------
    # Converting suggestions → ERP payload
    # ---------------------------

    def _compile_po_adjustments_payload(self, suggestions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Turn suggestion output into an ERP-adjustment payload.
        Only includes "safe" actions by default: qty_change, expedite, delivery date changes.
        """
        adjustments = []

        for po_s in suggestions.get("suggestions", []):
            po_number = po_s.get("po_number")
            changes = []

            for line in po_s.get("line_suggestions", []):
                for act in line.get("recommended_actions", []):
                    t = act.get("type")

                    if t == "qty_change":
                        changes.append({
                            "type": "qty_change",
                            "line_id": act["line_id"],
                            "new_quantity": act["new_quantity"],
                        })

                    elif t == "expedite":
                        # applying expedite as price premium + note
                        changes.append({
                            "type": "expedite",
                            "line_id": act["line_id"],
                            "premium_pct": 0.15,
                        })

                    elif t == "delivery_date_push":
                        # Here we don’t compute exact date; leave for human or implement date logic if you want.
                        # Keeping safe default: DO NOT auto-push dates unless you implement exact date update.
                        pass

                    elif t == "split_order":
                        # splitting creates complexity → avoid auto-apply by default
                        pass

                    elif t == "supplier_swap_suggestion":
                        # supplier swap should go through planning optimizer + approvals
                        pass

            if changes:
                adjustments.append({"po_number": po_number, "changes": changes})

        return {"adjustments": adjustments}

    # ---------------------------
    # Summaries + Follow-ups
    # ---------------------------

    def _summarize(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(results)
        completed = sum(1 for r in results if r["status"] == ExecutionStatus.COMPLETED.value)
        failed = sum(1 for r in results if r["status"] == ExecutionStatus.FAILED.value)
        waiting = sum(1 for r in results if r["status"] == ExecutionStatus.WAITING_APPROVAL.value)
        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "waiting_approval": waiting,
            "success_rate": (completed / total) if total else 0.0,
        }

    async def _create_follow_up_tasks(self, company: Company, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        tasks = []
        for r in results:
            if r["status"] == ExecutionStatus.WAITING_APPROVAL.value:
                tasks.append({
                    "type": "approval_required",
                    "action_id": r.get("action_id"),
                    "execution_id": r.get("execution_id"),
                    "due": datetime.utcnow() + timedelta(days=2),
                    "owner": "procurement_lead",
                })
            if r["status"] == ExecutionStatus.FAILED.value:
                tasks.append({
                    "type": "remediation",
                    "action_id": r.get("action_id"),
                    "execution_id": r.get("execution_id"),
                    "due": datetime.utcnow() + timedelta(days=1),
                    "owner": "system_admin",
                    "error": r.get("error"),
                })
        return tasks