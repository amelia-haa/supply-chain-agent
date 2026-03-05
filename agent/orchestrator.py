from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from agent.tools import (
    build_business_impact_report,
    build_judging_scorecard,
    build_uncertainty_bands,
    apply_proactive_triggers,
    build_executive_summary,
    detect_signal_drift,
    generate_playbook_autopilot,
    optimize_supplier_portfolio,
    build_responsible_ai_report,
    build_cost_value_report,
    derive_memory_feedback,
    estimate_mitigation_success_score,
    generate_actions,
    get_company_profile,
    log_mock_workflow_execution,
    read_memory,
    run_cost_optimized_pipeline,
    score_risk,
    simulate_tradeoffs,
    write_memory,
)


@dataclass
class CycleResult:
    timestamp_utc: str
    company: Dict[str, Any]
    events: List[Dict[str, Any]]
    memory_feedback: Dict[str, Any]
    pipeline_stats: Dict[str, Any]
    risk: Dict[str, Any]
    plan: List[Dict[str, Any]]
    actions: Dict[str, Any]
    cost_value_report: Dict[str, Any]
    business_impact_report: Dict[str, Any]
    judging_scorecard: Dict[str, Any]
    uncertainty_bands: Dict[str, Any]
    portfolio_optimization: Dict[str, Any]
    playbook_autopilot: Dict[str, Any]
    drift_report: Dict[str, Any]
    executive_summary: Dict[str, Any]
    responsible_ai_report: Dict[str, Any]
    workflow_execution_log: Dict[str, Any]
    autonomous_decision: Dict[str, Any]
    transparency_trace: Dict[str, Any]
    memory_write: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_utc": self.timestamp_utc,
            "company": self.company,
            "events": self.events,
            "memory_feedback": self.memory_feedback,
            "pipeline_stats": self.pipeline_stats,
            "risk": self.risk,
            "plan": self.plan,
            "actions": self.actions,
            "cost_value_report": self.cost_value_report,
            "business_impact_report": self.business_impact_report,
            "judging_scorecard": self.judging_scorecard,
            "uncertainty_bands": self.uncertainty_bands,
            "portfolio_optimization": self.portfolio_optimization,
            "playbook_autopilot": self.playbook_autopilot,
            "drift_report": self.drift_report,
            "executive_summary": self.executive_summary,
            "responsible_ai_report": self.responsible_ai_report,
            "workflow_execution_log": self.workflow_execution_log,
            "autonomous_decision": self.autonomous_decision,
            "transparency_trace": self.transparency_trace,
            "memory_write": self.memory_write,
        }


class AutonomousSupplyChainOrchestrator:
    """
    End-to-end orchestrator for a single autonomous resilience cycle.
    """

    def run_cycle(
        self,
        company_profile: Optional[Dict[str, Any]] = None,
        company_id: str = "de_semiconductor_auto",
    ) -> CycleResult:
        company = company_profile or get_company_profile(company_id=company_id)
        memory_feedback = derive_memory_feedback(company.get("company_name", "unknown"))
        optimized_pipeline = run_cost_optimized_pipeline(company)
        events = optimized_pipeline["events_for_risk"]
        pipeline_stats = optimized_pipeline["pipeline_stats"]
        risk = score_risk(company, events, memory_feedback=memory_feedback)
        plan = simulate_tradeoffs(company, risk)
        actions = generate_actions(company, risk, plan)
        cost_value_report = build_cost_value_report(risk, pipeline_stats, company)
        actions = apply_proactive_triggers(company, events, risk, cost_value_report, actions)
        responsible_ai_report = build_responsible_ai_report(company, risk, plan, events, actions)
        business_impact_report = build_business_impact_report(company, risk, plan, actions, cost_value_report)
        uncertainty_bands = build_uncertainty_bands(risk, cost_value_report)
        portfolio_optimization = optimize_supplier_portfolio(company)
        playbook_autopilot = generate_playbook_autopilot(events, risk, company)
        drift_report = detect_signal_drift(events)
        workflow_execution_log = log_mock_workflow_execution(company, risk, actions)
        mitigation_success_score = estimate_mitigation_success_score(risk, actions, cost_value_report)
        autonomous_decision = self._decide_autonomous_execution(risk, actions)
        transparency_trace = self._build_transparency_trace(
            company,
            events,
            pipeline_stats,
            risk,
            plan,
            actions,
            autonomous_decision,
            cost_value_report,
            responsible_ai_report,
        )

        memory_event = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "company_id": company.get("company_id"),
            "company_name": company.get("company_name"),
            "top_event": events[0] if events else None,
            "pipeline_stats": pipeline_stats,
            "risk": risk,
            "top_action": actions.get("recommended_top_action"),
            "cost_value_report": cost_value_report,
            "business_impact_report": business_impact_report,
            "responsible_ai_report": responsible_ai_report,
            "mitigation_success_score": mitigation_success_score,
            "supplier_health_score_current": actions.get("supplier_health_score_current"),
            "workflow_execution_log": workflow_execution_log,
            "approval_required": actions.get("human_approval_required", False),
            "autonomous_execution": autonomous_decision,
        }
        memory_write = write_memory(memory_event)
        provisional_result = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "company": company,
            "events": events,
            "memory_feedback": memory_feedback,
            "pipeline_stats": pipeline_stats,
            "risk": risk,
            "plan": plan,
            "actions": actions,
            "cost_value_report": cost_value_report,
            "business_impact_report": business_impact_report,
            "responsible_ai_report": responsible_ai_report,
            "uncertainty_bands": uncertainty_bands,
            "portfolio_optimization": portfolio_optimization,
            "playbook_autopilot": playbook_autopilot,
            "drift_report": drift_report,
            "workflow_execution_log": workflow_execution_log,
            "autonomous_decision": autonomous_decision,
            "transparency_trace": transparency_trace,
            "memory_write": memory_write,
        }
        judging_scorecard = build_judging_scorecard(provisional_result)
        executive_summary = build_executive_summary(provisional_result)

        return CycleResult(
            timestamp_utc=provisional_result["timestamp_utc"],
            company=company,
            events=events,
            memory_feedback=memory_feedback,
            pipeline_stats=pipeline_stats,
            risk=risk,
            plan=plan,
            actions=actions,
            cost_value_report=cost_value_report,
            business_impact_report=business_impact_report,
            judging_scorecard=judging_scorecard,
            uncertainty_bands=uncertainty_bands,
            portfolio_optimization=portfolio_optimization,
            playbook_autopilot=playbook_autopilot,
            drift_report=drift_report,
            executive_summary=executive_summary,
            responsible_ai_report=responsible_ai_report,
            workflow_execution_log=workflow_execution_log,
            autonomous_decision=autonomous_decision,
            transparency_trace=transparency_trace,
            memory_write=memory_write,
        )

    def memory(self) -> Dict[str, Any]:
        return read_memory()

    def _decide_autonomous_execution(self, risk: Dict[str, Any], actions: Dict[str, Any]) -> Dict[str, Any]:
        risk_score = float(risk.get("risk_score", 0.0))
        execution_mode = str(actions.get("execution_mode", "dry_run"))
        if execution_mode == "auto_execute" and risk_score >= 0.78:
            return {
                "mode": "auto_with_human_oversight",
                "status": "triggered",
                "reason": "Risk score exceeded autonomous threshold and policy enabled auto execution.",
            }
        return {
            "mode": "human_review",
            "status": "queued_for_approval",
            "reason": "Policy requires review/dry-run for this risk level.",
        }

    def _build_transparency_trace(
        self,
        company: Dict[str, Any],
        events: List[Dict[str, Any]],
        pipeline_stats: Dict[str, Any],
        risk: Dict[str, Any],
        plan: List[Dict[str, Any]],
        actions: Dict[str, Any],
        autonomous_decision: Dict[str, Any],
        cost_value_report: Dict[str, Any],
        responsible_ai_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "stage_sequence": ["perception", "risk_intelligence", "planning", "action", "memory"],
            "key_inputs": {
                "company_id": company.get("company_id"),
                "event_count": len(events),
                "pipeline_stats": pipeline_stats,
                "risk_components": risk.get("components", {}),
            },
            "decision_logic": {
                "risk_level": risk.get("risk_level"),
                "risk_score": risk.get("risk_score"),
                "early_warning_detected": risk.get("early_warning_detected", False),
                "top_plan": plan[0]["action"] if plan else None,
                "human_approval_required": actions.get("human_approval_required", False),
                "execution_mode": actions.get("execution_mode"),
                "autonomy_policy": actions.get("autonomy_policy"),
                "autonomous_mode": autonomous_decision.get("mode"),
                "tiered_alert_action": actions.get("tiered_alert_action"),
                "guardrail_flags": actions.get("guardrail_flags", []),
            },
            "cost_value_summary": cost_value_report,
            "responsible_ai_controls": {
                "human_in_the_loop": True,
                "override_threshold": "risk_score >= 0.78 triggers autonomous workflow with approval gate",
                "bias_check_status": responsible_ai_report.get("bias_check_status", "unknown"),
                "validation_status": responsible_ai_report.get("status", "unknown"),
                "checks": responsible_ai_report.get("checks", []),
                "findings": responsible_ai_report.get("findings", []),
            },
        }
