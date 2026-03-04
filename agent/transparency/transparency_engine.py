# src/transparency/transparency_engine.py
from datetime import datetime
from typing import Dict, Any, List, Optional
from agent.models import Company, Supplier
from .trace_types import ReasonTrace, ConstraintViolation
from .risk_justifier import RiskJustifier
from .override_policy import OverridePolicy
from .validators import ConstraintValidator, BiasValidator

class TransparencyEngine:
    """
    Produces a full transparency package:
      - structured reasoning trace
      - risk justification
      - constraint + bias validation
      - override mode (automatic/human/executive)
    """

    def __init__(self):
        self.risk_justifier = RiskJustifier()
        self.override_policy = OverridePolicy()
        self.constraint_validator = ConstraintValidator()
        self.bias_validator = BiasValidator()

    async def build_planning_trace(
        self,
        company: Company,
        suppliers: List[Supplier],
        scenario_summary: Dict[str, Any],
        allocation_plan: Dict[str, Any],
        buffer_policy: Dict[str, Any],
        decision_tree_result: Dict[str, Any],
        estimated_cost: float = 0.0,
        recommendation_score: float = 0.75
    ) -> ReasonTrace:
        trace = ReasonTrace(
            trace_id=f"TRACE_{int(datetime.utcnow().timestamp())}",
            created_at=datetime.utcnow().isoformat(),
            decision_stage="planning",
            decision_name="Comprehensive Planning Decision",
            inputs_summary={
                "company": {"id": company.id, "risk_appetite": company.risk_appetite, "service_target": company.service_level_target},
                "scenario": {"service": scenario_summary.get("service"), "cost_multiplier": scenario_summary.get("cost_multiplier")},
            }
        )

        trace.add_step("Scenario evaluation", {"scenario_summary": scenario_summary})
        trace.add_step("Supplier reallocation", {"allocation_kpis": allocation_plan.get("kpis"), "allocations": allocation_plan.get("allocations")})
        trace.add_step("Buffer stock modeling", {"buffer_outputs": buffer_policy.get("outputs"), "tradeoff": buffer_policy.get("cost_tradeoff")})
        trace.add_step("Decision tree reasoning", {"decision_tree": decision_tree_result})

        # Risk justification
        risk_just = await self.risk_justifier.justify_company_risk(company, suppliers, scenario_summary)
        trace.risk_justification = risk_just

        # Constraints
        trace.constraints_checked = ["allocation_concentration", "capacity_feasibility", "budget_guardrail"]
        violations = []
        violations.extend(await self.constraint_validator.validate_allocation_constraints(
            allocation_plan,
            max_single_supplier_pct=self.override_policy.max_single_supplier_allocation_pct
        ))
        violations.extend(await self.constraint_validator.validate_budget_constraint(
            estimated_cost=estimated_cost,
            annual_revenue=getattr(company, "annual_revenue", 0.0),
            max_fraction=self.override_policy.max_budget_fraction_of_revenue
        ))
        for v in violations:
            trace.add_violation(v)

        # Bias checks (supply-base bias)
        trace.bias_checks = ["region_balance", "region_exclusion"]
        trace.bias_findings = await self.bias_validator.validate_supplier_region_balance(suppliers, allocation_plan)

        # Override decision
        has_high_violation = any(v.severity in ["high", "critical"] for v in trace.constraint_violations)
        service_p05 = scenario_summary.get("service", {}).get("p05", company.service_level_target)
        mode = self.override_policy.decide_approval_mode(
            recommendation_score=recommendation_score,
            has_high_severity_violation=has_high_violation,
            estimated_cost=estimated_cost,
            annual_revenue=getattr(company, "annual_revenue", 0.0),
            service_p05=service_p05
        )

        trace.override_policy = mode
        trace.override_required = mode["mode"] != "automatic"
        trace.override_reason = mode["reason"]

        # Final decision summary
        trace.final_decision = {
            "approval_mode": mode["mode"],
            "reason": mode["reason"],
            "recommendation_score": recommendation_score
        }

        return trace
