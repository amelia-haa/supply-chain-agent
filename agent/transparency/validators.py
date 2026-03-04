# src/transparency/validators.py
from typing import Dict, Any, List
import numpy as np
from agent.models.supply_chain import Company, Supplier
from .trace_types import ConstraintViolation

class ConstraintValidator:
    """
    Constraint validation:
      - allocation concentration caps
      - budget caps
      - non-negative quantities
      - capacity feasibility
    """

    async def validate_allocation_constraints(
        self,
        allocation_plan: Dict[str, Any],
        max_single_supplier_pct: float = 0.70
    ) -> List[ConstraintViolation]:
        violations: List[ConstraintViolation] = []
        allocs = allocation_plan.get("allocations", [])

        for a in allocs:
            pct = float(a.get("allocation_pct", 0))
            if pct > max_single_supplier_pct:
                violations.append(ConstraintViolation(
                    code="ALLOC_CONCENTRATION",
                    severity="high",
                    message=f"Single supplier allocation {pct:.2f} exceeds cap {max_single_supplier_pct:.2f}.",
                    context={"supplier_id": a.get("supplier_id"), "allocation_pct": pct}
                ))

            units = float(a.get("units", 0))
            cap = float(a.get("capacity", 0))
            if cap > 0 and units > cap:
                violations.append(ConstraintViolation(
                    code="CAPACITY_EXCEEDED",
                    severity="critical",
                    message="Allocated units exceed supplier capacity.",
                    context={"supplier_id": a.get("supplier_id"), "units": units, "capacity": cap}
                ))

            if units < 0:
                violations.append(ConstraintViolation(
                    code="NEGATIVE_UNITS",
                    severity="critical",
                    message="Negative allocation units not allowed.",
                    context={"supplier_id": a.get("supplier_id"), "units": units}
                ))

        return violations

    async def validate_budget_constraint(
        self,
        estimated_cost: float,
        annual_revenue: float,
        max_fraction: float = 0.05
    ) -> List[ConstraintViolation]:
        violations: List[ConstraintViolation] = []
        if annual_revenue > 0 and estimated_cost > max_fraction * annual_revenue:
            violations.append(ConstraintViolation(
                code="BUDGET_GUARDRAIL",
                severity="high",
                message=f"Estimated cost {estimated_cost:.0f} exceeds {max_fraction*100:.1f}% of revenue.",
                context={"estimated_cost": estimated_cost, "annual_revenue": annual_revenue}
            ))
        return violations

class BiasValidator:
    """
    Bias checks (practical version):
      - detect systematically excluding regions without risk evidence
      - detect disproportionate penalty on a region compared to others
    NOTE: this is not “fairness” in a human-demographics sense—here it's supply-base bias.
    """

    async def validate_supplier_region_balance(
        self,
        suppliers: List[Supplier],
        allocation_plan: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        allocs = allocation_plan.get("allocations", [])
        if not allocs:
            return []

        supplier_by_id = {s.id: s for s in suppliers}

        # region allocations
        region_alloc = {}
        for a in allocs:
            sid = a.get("supplier_id")
            s = supplier_by_id.get(sid)
            if not s:
                continue
            region_alloc.setdefault(s.region, 0.0)
            region_alloc[s.region] += float(a.get("allocation_pct", 0))

        regions = list(region_alloc.keys())
        pct = np.array([region_alloc[r] for r in regions], dtype=float)

        findings = []

        # If one region receives near-zero allocation, check if its risk is actually high
        for r in regions:
            if region_alloc[r] < 0.05:
                # average risk of region suppliers
                r_suppliers = [s for s in suppliers if s.region == r]
                if r_suppliers:
                    avg_risk = float(np.mean([s.risk_score for s in r_suppliers]))
                    findings.append({
                        "check": "REGION_EXCLUSION",
                        "region": r,
                        "allocation_pct": region_alloc[r],
                        "avg_region_risk": avg_risk,
                        "note": "If region is excluded, ensure risk evidence supports it."
                    })

        # dominance check
        if pct.max() > 0.90:
            findings.append({
                "check": "REGION_DOMINANCE",
                "region": regions[int(pct.argmax())],
                "allocation_pct": float(pct.max()),
                "note": "Allocation heavily concentrated in one region; increases systemic regional risk."
            })

        return findings