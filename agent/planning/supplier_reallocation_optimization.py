"""
Supplier Reallocation Optimization (Planning & Decision Engine)

What this does (real-world shape):
- Given demand per component/SKU and a set of suppliers (with cost, capacity, lead time, risk, reliability/quality),
  compute an optimal reallocation (how much to buy from each supplier) that:
    1) meets demand for each component
    2) respects supplier capacity constraints
    3) minimizes a weighted objective: total cost + risk + lead-time penalty - reliability/quality reward

Implementation details:
- Primary solver: scipy.optimize.linprog (Linear Programming) if SciPy is available
- Fallback solver: deterministic greedy heuristic if SciPy is not available or LP fails
- Fully async-friendly (all public methods are async), so it fits your agent architecture
- Accepts Supplier objects OR dicts (robust field extraction with defaults)

You can drop this file as:
  src/planning/supplier_reallocation_optimizer.py
and import it in your PlanningEngine.

Requirements (recommended):
  pip install scipy
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import logging

logger = logging.getLogger(__name__)

# Optional SciPy (LP solver). We gracefully fallback if not installed.
try:
    from scipy.optimize import linprog  # type: ignore
    _SCIPY_AVAILABLE = True
except Exception:
    linprog = None
    _SCIPY_AVAILABLE = False


# ----------------------------
# Helper: robust field access
# ----------------------------

def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Get attribute or dict key from obj."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, float(v)) for v in weights.values())
    if total <= 0:
        # safe defaults
        return {k: 1.0 / max(1, len(weights)) for k in weights}
    return {k: float(v) / total for k, v in weights.items()}


# ----------------------------
# Data models used internally
# ----------------------------

@dataclass(frozen=True)
class SupplierProfile:
    supplier_id: str
    name: str
    region: str

    # per-component dictionaries
    unit_cost: Dict[str, float]              # cost per unit for component
    max_capacity: Dict[str, float]           # max units for component over horizon
    lead_time_days: Dict[str, float]         # lead time per component

    # scalar scores (0..1) where higher is better for reliability/quality
    reliability: float
    quality: float

    # scalar risk score (0..1) where higher is worse
    risk: float


def _build_supplier_profile(supplier: Any) -> SupplierProfile:
    """
    Convert your Supplier object/dict to a normalized SupplierProfile.

    Expected (but optional) fields on Supplier:
      - id (str), name (str), region/location (str)
      - unit_costs or cost_per_unit: dict {component: cost}
      - max_capacity or capacities: dict {component: capacity_units}
      - lead_time_days or lead_times: dict {component: days}
      - reliability_score (0..1), quality_score (0..1), risk_score (0..1)

    Missing fields are filled with safe defaults.
    """
    sid = str(_get(supplier, "id", _get(supplier, "supplier_id", "unknown")))
    name = str(_get(supplier, "name", sid))
    region = str(_get(supplier, "region", _get(supplier, "location", "unknown")))

    # cost dict
    unit_cost = _get(supplier, "unit_costs", None)
    if unit_cost is None:
        unit_cost = _get(supplier, "cost_per_unit", None)
    if unit_cost is None:
        unit_cost = _get(supplier, "costs", {})
    if not isinstance(unit_cost, dict):
        unit_cost = {}

    # capacity dict
    max_capacity = _get(supplier, "max_capacity", None)
    if max_capacity is None:
        max_capacity = _get(supplier, "capacities", None)
    if max_capacity is None:
        max_capacity = _get(supplier, "capacity", {})
    if not isinstance(max_capacity, dict):
        max_capacity = {}

    # lead time dict
    lead_time = _get(supplier, "lead_time_days", None)
    if lead_time is None:
        lead_time = _get(supplier, "lead_times", None)
    if lead_time is None:
        lead_time = _get(supplier, "lead_time", {})
    if not isinstance(lead_time, dict):
        lead_time = {}

    reliability = _clamp(_as_float(_get(supplier, "reliability_score", _get(supplier, "reliability", 0.7)), 0.7))
    quality = _clamp(_as_float(_get(supplier, "quality_score", _get(supplier, "quality", 0.7)), 0.7))
    risk = _clamp(_as_float(_get(supplier, "risk_score", _get(supplier, "risk", 0.3)), 0.3))

    # sanitize dict values
    unit_cost = {str(k): max(0.0, _as_float(v, 0.0)) for k, v in unit_cost.items()}
    max_capacity = {str(k): max(0.0, _as_float(v, 0.0)) for k, v in max_capacity.items()}
    lead_time = {str(k): max(0.0, _as_float(v, 0.0)) for k, v in lead_time.items()}

    return SupplierProfile(
        supplier_id=sid,
        name=name,
        region=region,
        unit_cost=unit_cost,
        max_capacity=max_capacity,
        lead_time_days=lead_time,
        reliability=reliability,
        quality=quality,
        risk=risk,
    )


# ----------------------------
# Main Optimizer
# ----------------------------

class SupplierReallocationOptimizer:
    """
    Supplier Reallocation Optimization:
    - Inputs:
        company: Company object (used only for metadata; not required)
        suppliers: list[Supplier] (object or dict)
        demand: dict {component: demand_units}  (required)
    - Output:
        allocation plan with per-component supplier splits, cost, risk, and service metrics.
    """

    def __init__(self) -> None:
        # Objective weights (normalized internally)
        # Increase cost weight if you care more about spending,
        # increase risk/lead_time weights if you care more about resilience/service.
        self.objective_weights = _normalize_weights({
            "cost": 0.45,
            "risk": 0.25,
            "lead_time": 0.20,
            "reliability_reward": 0.05,
            "quality_reward": 0.05,
        })

        # Soft penalty if an allocation exceeds a target lead time (service constraint)
        self.service_level = {
            "target_lead_time_days": 21.0,     # typical target; override in optimize() if you want
            "late_penalty_per_day": 0.02,      # penalty multiplier (dimensionless)
        }

    async def optimize_supplier_reallocation(
        self,
        company: Any,
        suppliers: List[Any],
        demand_by_component: Dict[str, float],
        *,
        constraints: Optional[Dict[str, Any]] = None,
        objective_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Optimize supplier reallocation.

        constraints (optional):
          - max_total_suppliers_per_component: int (e.g., 3)  [heuristic/fallback uses it; LP ignores unless you add integer programming]
          - target_lead_time_days: float
          - min_reliability: float (0..1)  [enforced as feasibility by filtering suppliers]
          - max_risk: float (0..1)         [enforced as feasibility by filtering suppliers]
          - allow_partial_fulfillment: bool (default False)
        """
        t0 = datetime.utcnow()
        constraints = constraints or {}
        allow_partial = bool(constraints.get("allow_partial_fulfillment", False))

        weights = self.objective_weights if objective_weights is None else _normalize_weights(objective_weights)
        target_lt = float(constraints.get("target_lead_time_days", self.service_level["target_lead_time_days"]))
        late_penalty = float(constraints.get("late_penalty_per_day", self.service_level["late_penalty_per_day"]))

        # Build supplier profiles
        profiles = [_build_supplier_profile(s) for s in suppliers]

        # Basic input validation
        demand = {str(k): max(0.0, _as_float(v, 0.0)) for k, v in demand_by_component.items()}
        demand = {k: v for k, v in demand.items() if v > 0.0}
        if not demand:
            return {
                "company_id": str(_get(company, "id", "unknown")),
                "timestamp": datetime.utcnow(),
                "status": "no_demand",
                "message": "No positive demand provided.",
                "allocations": {},
            }

        # Filter suppliers by hard constraints (optional)
        min_rel = constraints.get("min_reliability", None)
        max_risk = constraints.get("max_risk", None)
        if min_rel is not None:
            min_rel = float(min_rel)
            profiles = [p for p in profiles if p.reliability >= min_rel]
        if max_risk is not None:
            max_risk = float(max_risk)
            profiles = [p for p in profiles if p.risk <= max_risk]

        if not profiles:
            return {
                "company_id": str(_get(company, "id", "unknown")),
                "timestamp": datetime.utcnow(),
                "status": "no_suppliers",
                "message": "No suppliers remain after applying constraints (min_reliability/max_risk).",
                "allocations": {},
            }

        # Try LP first (best for real-world continuous allocation)
        plan = None
        if _SCIPY_AVAILABLE:
            try:
                plan = await self._solve_with_linear_programming(
                    profiles, demand,
                    weights=weights,
                    target_lead_time_days=target_lt,
                    late_penalty_per_day=late_penalty,
                    allow_partial_fulfillment=allow_partial,
                )
            except Exception as e:
                logger.warning(f"LP optimization failed, will fallback to greedy. Error={e}")

        # Fallback greedy if LP not available or failed
        if plan is None:
            plan = await self._solve_with_greedy_heuristic(
                profiles, demand,
                weights=weights,
                target_lead_time_days=target_lt,
                late_penalty_per_day=late_penalty,
                max_suppliers_per_component=int(constraints.get("max_total_suppliers_per_component", 3)),
                allow_partial_fulfillment=allow_partial,
            )

        # Add summary metrics
        summary = self._summarize_plan(plan, demand)

        return {
            "company_id": str(_get(company, "id", "unknown")),
            "timestamp": datetime.utcnow(),
            "status": plan["status"],
            "solver": plan["solver"],
            "objective_weights": weights,
            "service_targets": {"target_lead_time_days": target_lt, "late_penalty_per_day": late_penalty},
            "allocations": plan["allocations"],          # per component splits
            "unfilled_demand": plan.get("unfilled_demand", {}),
            "summary": summary,
            "runtime_ms": int((datetime.utcnow() - t0).total_seconds() * 1000),
        }

    # ----------------------------
    # LP Solver (SciPy linprog)
    # ----------------------------

    async def _solve_with_linear_programming(
        self,
        suppliers: List[SupplierProfile],
        demand: Dict[str, float],
        *,
        weights: Dict[str, float],
        target_lead_time_days: float,
        late_penalty_per_day: float,
        allow_partial_fulfillment: bool,
    ) -> Dict[str, Any]:
        """
        Linear programming formulation:
        Variables: x_{s,c} = units allocated from supplier s for component c  (continuous >=0)
        Constraints:
          - For each component c: sum_s x_{s,c} == demand[c] (or <= if allow_partial_fulfillment)
          - For each supplier s and component c: x_{s,c} <= capacity_{s,c}
        Objective:
          minimize sum_{s,c} x_{s,c} * unit_objective_cost(s,c)

        unit_objective_cost(s,c) combines:
          cost, risk, lead time penalty, reliability reward, quality reward
        """
        components = list(demand.keys())
        S = len(suppliers)
        C = len(components)

        # Build variable index mapping (s,c) -> idx
        idx = {}
        k = 0
        for si in range(S):
            for ci in range(C):
                idx[(si, ci)] = k
                k += 1
        n_vars = S * C

        # Objective vector
        cvec = np.zeros(n_vars, dtype=float)

        # Bounds: 0 <= x_{s,c} <= capacity_{s,c}
        bounds: List[Tuple[float, float]] = []
        for si, sp in enumerate(suppliers):
            for ci, comp in enumerate(components):
                cap = sp.max_capacity.get(comp, 0.0)
                # If supplier doesn't support the component, cap=0 forces x=0.
                ub = float(cap) if cap > 0 else 0.0
                bounds.append((0.0, ub))

                unit_cost = sp.unit_cost.get(comp, np.inf)
                if not np.isfinite(unit_cost):
                    # if cost missing but cap>0, assign a big cost so it will be avoided
                    unit_cost = 1e9

                lt = sp.lead_time_days.get(comp, target_lead_time_days)
                late_days = max(0.0, float(lt) - float(target_lead_time_days))
                lead_pen = late_days * float(late_penalty_per_day)

                # Note: risk is "bad" (higher worse), reliability/quality are "good"
                unit_obj = (
                    weights["cost"] * float(unit_cost)
                    + weights["risk"] * float(sp.risk) * 1000.0  # scale risk into money-like penalty
                    + weights["lead_time"] * float(lead_pen) * 1000.0
                    - weights["reliability_reward"] * float(sp.reliability) * 200.0
                    - weights["quality_reward"] * float(sp.quality) * 200.0
                )

                cvec[idx[(si, ci)]] = unit_obj

        # Equality/inequality constraints for demand fulfillment
        A_eq = []
        b_eq = []
        A_ub = []
        b_ub = []

        for ci, comp in enumerate(components):
            row = np.zeros(n_vars, dtype=float)
            for si in range(S):
                row[idx[(si, ci)]] = 1.0

            if allow_partial_fulfillment:
                # sum_s x_{s,c} <= demand[c]
                A_ub.append(row)
                b_ub.append(float(demand[comp]))
            else:
                # sum_s x_{s,c} == demand[c]
                A_eq.append(row)
                b_eq.append(float(demand[comp]))

        # Solve
        res = linprog(
            c=cvec,
            A_ub=np.array(A_ub) if A_ub else None,
            b_ub=np.array(b_ub) if b_ub else None,
            A_eq=np.array(A_eq) if A_eq else None,
            b_eq=np.array(b_eq) if b_eq else None,
            bounds=bounds,
            method="highs",
        )

        if not res.success:
            raise RuntimeError(f"linprog failed: {res.message}")

        x = res.x  # optimal allocations

        allocations: Dict[str, List[Dict[str, Any]]] = {comp: [] for comp in components}
        unfilled: Dict[str, float] = {}

        for ci, comp in enumerate(components):
            comp_demand = float(demand[comp])
            supplied = 0.0
            for si, sp in enumerate(suppliers):
                units = float(x[idx[(si, ci)]])
                if units <= 1e-9:
                    continue
                supplied += units

                unit_cost = sp.unit_cost.get(comp, 0.0)
                lt = sp.lead_time_days.get(comp, target_lead_time_days)

                allocations[comp].append({
                    "supplier_id": sp.supplier_id,
                    "supplier_name": sp.name,
                    "region": sp.region,
                    "units": units,
                    "share": units / comp_demand if comp_demand > 0 else 0.0,
                    "unit_cost": unit_cost,
                    "extended_cost": units * unit_cost,
                    "lead_time_days": lt,
                    "risk_score": sp.risk,
                    "reliability": sp.reliability,
                    "quality": sp.quality,
                })

            if allow_partial_fulfillment and supplied + 1e-6 < comp_demand:
                unfilled[comp] = comp_demand - supplied

            # sort per component by units desc
            allocations[comp].sort(key=lambda r: r["units"], reverse=True)

        status = "ok" if not unfilled else "partial_fulfillment"

        return {
            "status": status,
            "solver": "scipy.linprog",
            "allocations": allocations,
            "unfilled_demand": unfilled,
        }

    # ----------------------------
    # Greedy fallback solver
    # ----------------------------

    async def _solve_with_greedy_heuristic(
        self,
        suppliers: List[SupplierProfile],
        demand: Dict[str, float],
        *,
        weights: Dict[str, float],
        target_lead_time_days: float,
        late_penalty_per_day: float,
        max_suppliers_per_component: int,
        allow_partial_fulfillment: bool,
    ) -> Dict[str, Any]:
        """
        Greedy heuristic:
        For each component:
          - compute a "score per unit" for each supplier that can supply it
          - allocate demand to best suppliers until demand is satisfied or capacities exhausted

        This is not as optimal as LP, but it is:
          - fast
          - deterministic
          - works without SciPy
        """
        allocations: Dict[str, List[Dict[str, Any]]] = {}
        unfilled: Dict[str, float] = {}

        for comp, comp_demand in demand.items():
            comp_demand = float(comp_demand)
            remaining = comp_demand
            options = []

            for sp in suppliers:
                cap = float(sp.max_capacity.get(comp, 0.0))
                if cap <= 0:
                    continue

                unit_cost = sp.unit_cost.get(comp, None)
                if unit_cost is None:
                    # missing cost -> treat as very expensive
                    unit_cost = 1e9
                unit_cost = float(unit_cost)

                lt = float(sp.lead_time_days.get(comp, target_lead_time_days))
                late_days = max(0.0, lt - float(target_lead_time_days))
                lead_pen = late_days * float(late_penalty_per_day)

                unit_obj = (
                    weights["cost"] * unit_cost
                    + weights["risk"] * sp.risk * 1000.0
                    + weights["lead_time"] * lead_pen * 1000.0
                    - weights["reliability_reward"] * sp.reliability * 200.0
                    - weights["quality_reward"] * sp.quality * 200.0
                )

                options.append((unit_obj, sp, cap, unit_cost, lt))

            options.sort(key=lambda t: t[0])  # lower objective is better

            allocations[comp] = []
            used_suppliers = 0

            for unit_obj, sp, cap, unit_cost, lt in options:
                if remaining <= 1e-9:
                    break
                if used_suppliers >= max_suppliers_per_component:
                    break

                units = min(remaining, cap)
                remaining -= units
                used_suppliers += 1

                allocations[comp].append({
                    "supplier_id": sp.supplier_id,
                    "supplier_name": sp.name,
                    "region": sp.region,
                    "units": units,
                    "share": units / comp_demand if comp_demand > 0 else 0.0,
                    "unit_cost": unit_cost,
                    "extended_cost": units * unit_cost,
                    "lead_time_days": lt,
                    "risk_score": sp.risk,
                    "reliability": sp.reliability,
                    "quality": sp.quality,
                    "heuristic_unit_objective": unit_obj,
                })

            if remaining > 1e-6:
                if allow_partial_fulfillment:
                    unfilled[comp] = remaining
                else:
                    # In strict mode, greedy failing means infeasible with given capacities.
                    # We still return what we could allocate, but mark as infeasible.
                    unfilled[comp] = remaining

        status = "ok" if not unfilled else ("partial_fulfillment" if allow_partial_fulfillment else "infeasible_capacity")
        return {
            "status": status,
            "solver": "greedy.fallback",
            "allocations": allocations,
            "unfilled_demand": unfilled,
        }

    # ----------------------------
    # Summary metrics
    # ----------------------------

    def _summarize_plan(self, plan: Dict[str, Any], demand: Dict[str, float]) -> Dict[str, Any]:
        allocations = plan["allocations"]
        total_cost = 0.0
        total_units = 0.0

        # Weighted averages by units
        risk_num = 0.0
        lt_num = 0.0
        rel_num = 0.0
        qual_num = 0.0

        for comp, rows in allocations.items():
            for r in rows:
                units = float(r["units"])
                total_units += units
                total_cost += float(r.get("extended_cost", 0.0))
                risk_num += units * float(r.get("risk_score", 0.0))
                lt_num += units * float(r.get("lead_time_days", 0.0))
                rel_num += units * float(r.get("reliability", 0.0))
                qual_num += units * float(r.get("quality", 0.0))

        avg_risk = (risk_num / total_units) if total_units > 0 else 0.0
        avg_lt = (lt_num / total_units) if total_units > 0 else 0.0
        avg_rel = (rel_num / total_units) if total_units > 0 else 0.0
        avg_qual = (qual_num / total_units) if total_units > 0 else 0.0

        # fulfillment rate
        total_demand = sum(float(v) for v in demand.values())
        unfilled = plan.get("unfilled_demand", {})
        total_unfilled = sum(float(v) for v in unfilled.values()) if unfilled else 0.0
        fulfillment_rate = (1.0 - total_unfilled / total_demand) if total_demand > 0 else 1.0

        return {
            "total_demand_units": total_demand,
            "total_allocated_units": total_units,
            "fulfillment_rate": _clamp(fulfillment_rate, 0.0, 1.0),
            "total_cost": total_cost,
            "avg_risk_score": avg_risk,
            "avg_lead_time_days": avg_lt,
            "avg_reliability": avg_rel,
            "avg_quality": avg_qual,
            "notes": (
                "LP solver gives globally optimal continuous allocations; greedy is a safe fallback. "
                "If you need discrete constraints (e.g., at most K suppliers per component strictly), "
                "that becomes a mixed-integer optimization (MILP)."
            ),
        }


# ----------------------------
# Example usage (async)
# ----------------------------
# async def demo(company, suppliers):
#     optimizer = SupplierReallocationOptimizer()
#     demand = {"MCU": 10000, "SENSOR": 5000}
#     result = await optimizer.optimize_supplier_reallocation(company, suppliers, demand)
#     print(result)

# from src.planning.supplier_reallocation_optimizer import SupplierReallocationOptimizer

# optimizer = SupplierReallocationOptimizer()
# realloc_plan = await optimizer.optimize_supplier_reallocation(
#     company,
#     suppliers,
#     demand_by_component,   # real demand from ERP/MRP
#     constraints={
#         "target_lead_time_days": 21,
#         "min_reliability": 0.6,
#         "max_risk": 0.8,
#         "allow_partial_fulfillment": False
#     }
# )