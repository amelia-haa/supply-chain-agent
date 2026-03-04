"""
Buffer Stock Strategy Modeling (Planning & Decision Engine)

Goal (real-world):
- Decide safety stock / buffer stock levels per SKU/component to hit a target service level
  while minimizing total cost (holding + stockout) under demand & lead time uncertainty.

This module provides:
1) Classic safety stock (Normal approx):
   SS = z(service_level) * sqrt( LT * sigma_d^2 + (mu_d^2) * sigma_LT^2 )
2) Reorder point:
   ROP = mu_d * LT + SS
3) Simulation-based validation (Monte Carlo) to estimate achieved service level & expected cost
4) Scenario comparison (cost vs service trade-off) across multiple service level targets / policy multipliers

Async-friendly: all public methods are async.

Recommended deps:
  pip install numpy scipy
SciPy is optional but helps for accurate z-scores; fallback approximation included.
"""

from __future__ import annotations

from ast import Compare
import asyncio
from dataclasses import dataclass
from datetime import datetime
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import logging

logger = logging.getLogger(__name__)

# Optional SciPy for accurate inverse CDF (z-score)
try:
    from scipy.stats import norm  # type: ignore
    _SCIPY_STATS = True
except Exception:
    norm = None
    _SCIPY_STATS = False


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _z_from_service_level(service_level: float) -> float:
    """
    Convert cycle service level (CSL) to z-score.
    CSL is P(no stockout during lead time) for continuous review ROP policy.
    """
    sl = _clamp(float(service_level), 0.50, 0.9999)
    if _SCIPY_STATS:
        return float(norm.ppf(sl))

    # Fallback approximation for inverse normal CDF (Acklam-like simplified)
    # Good enough for planning; SciPy recommended for production.
    # Reference shape: https://www.johndcook.com/blog/normal_cdf_inverse/
    # (implemented here without external lookup)
    a1, a2, a3, a4, a5, a6 = -39.6968302866538, 220.946098424521, -275.928510446969, 138.357751867269, -30.6647980661472, 2.50662827745924
    b1, b2, b3, b4, b5 = -54.4760987982241, 161.585836858041, -155.698979859887, 66.8013118877197, -13.2806815528857
    c1, c2, c3, c4, c5, c6 = -0.00778489400243029, -0.322396458041136, -2.40075827716184, -2.54973253934373, 4.37466414146497, 2.93816398269878
    d1, d2, d3, d4 = 0.00778469570904146, 0.32246712907004, 2.445134137143, 3.75440866190742

    p = sl
    # break-points
    plow = 0.02425
    phigh = 1 - plow

    if p < plow:
        q = np.sqrt(-2 * np.log(p))
        return (((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) / ((((d1*q + d2)*q + d3)*q + d4)*q + 1)
    if p > phigh:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) / ((((d1*q + d2)*q + d3)*q + d4)*q + 1)

    q = p - 0.5
    r = q*q
    return (((((a1*r + a2)*r + a3)*r + a4)*r + a5)*r + a6)*q / (((((b1*r + b2)*r + b3)*r + b4)*r + b5)*r + 1)


@dataclass(frozen=True)
class ItemDemandStats:
    """
    Demand stats in consistent time unit (e.g., per day).
    mu_d: mean demand per day
    sigma_d: std dev of demand per day
    """
    mu_d: float
    sigma_d: float


@dataclass(frozen=True)
class LeadTimeStats:
    """
    Lead time stats in days.
    LT: mean lead time (days)
    sigma_LT: std dev lead time (days)
    """
    LT: float
    sigma_LT: float


class BufferStockModeler:
    def __init__(self) -> None:
        # Default cost parameters (override per item if you have real data)
        self.default_costs = {
            "holding_cost_rate_annual": 0.25,   # 25% of unit value per year
            "stockout_penalty_per_unit": 5.0,   # $ per unit short (placeholder)
            "days_per_year": 365.0,
        }

        # Monte Carlo parameters (for validation)
        self.simulation = {
            "runs": 5000,          # per item per scenario
            "random_seed": 42,
        }

    async def model_buffer_stock_strategy(
        self,
        company: Any,
        items: List[Dict[str, Any]],
        *,
        global_service_level_targets: Optional[List[float]] = None,
        policy_multipliers: Optional[List[float]] = None,
        validate_with_simulation: bool = True,
    ) -> Dict[str, Any]:
        """
        items: list of dicts, each item expects:
          required:
            - item_id (str)
            - mu_d (mean demand per day)
            - sigma_d (std demand per day)
            - LT (mean lead time days)
          optional:
            - sigma_LT (std lead time days) [default 0]
            - unit_cost ($) [used for holding cost]
            - holding_cost_rate_annual (override)
            - stockout_penalty_per_unit (override)
            - service_level_target (override per item)

        global_service_level_targets:
          scenarios to evaluate (e.g., [0.90, 0.95, 0.98])
        policy_multipliers:
          multiply computed safety stock by these (e.g., [0.8, 1.0, 1.2]) for trade-off curves

        Output:
          - per scenario: total cost, achieved service, and per-item SS/ROP
          - recommended scenario (min expected total cost that still meets service target)
        """
        t0 = datetime.utcnow()
        company_id = str(getattr(company, "id", "unknown"))

        service_targets = global_service_level_targets or [0.90, 0.95, 0.98]
        multipliers = policy_multipliers or [0.8, 1.0, 1.2]

        # Pre-parse items safely
        parsed_items = []
        for it in items:
            item_id = str(it.get("item_id", it.get("sku", "unknown")))
            mu_d = max(0.0, _as_float(it.get("mu_d"), 0.0))
            sigma_d = max(0.0, _as_float(it.get("sigma_d"), 0.0))
            LT = max(0.0, _as_float(it.get("LT"), 0.0))
            sigma_LT = max(0.0, _as_float(it.get("sigma_LT"), 0.0))

            if mu_d <= 0 or LT <= 0:
                # skip unusable item
                logger.warning(f"Skipping item {item_id}: mu_d and LT must be > 0 (got mu_d={mu_d}, LT={LT})")
                continue

            unit_cost = max(0.0, _as_float(it.get("unit_cost"), 0.0))
            h_rate = _as_float(it.get("holding_cost_rate_annual"), self.default_costs["holding_cost_rate_annual"])
            stockout_pen = _as_float(it.get("stockout_penalty_per_unit"), self.default_costs["stockout_penalty_per_unit"])

            item_sl = it.get("service_level_target", None)
            item_sl = float(item_sl) if item_sl is not None else None

            parsed_items.append({
                "item_id": item_id,
                "demand": ItemDemandStats(mu_d=mu_d, sigma_d=sigma_d),
                "lead_time": LeadTimeStats(LT=LT, sigma_LT=sigma_LT),
                "unit_cost": unit_cost,
                "holding_cost_rate_annual": h_rate,
                "stockout_penalty_per_unit": stockout_pen,
                "service_level_target": item_sl,
            })

        if not parsed_items:
            return {
                "company_id": company_id,
                "timestamp": datetime.utcnow(),
                "status": "no_items",
                "message": "No valid items to model (need mu_d>0 and LT>0).",
            }

        # Build scenarios: cartesian product of targets x multipliers
        scenarios = []
        for sl in service_targets:
            for m in multipliers:
                scenarios.append({"service_level": float(sl), "multiplier": float(m)})

        scenario_results = []
        for sc in scenarios:
            sc_result = await self._evaluate_scenario(parsed_items, sc["service_level"], sc["multiplier"], validate_with_simulation)
            scenario_results.append(sc_result)

        # Recommend: choose lowest total expected cost among those that meet target service
        feasible = [r for r in scenario_results if r["achieved_service_level"] >= r["target_service_level"] - 1e-6]
        if feasible:
            recommended = min(feasible, key=lambda r: r["total_expected_cost"])
        else:
            # If none meet, choose the best service level achieved
            recommended = max(scenario_results, key=lambda r: r["achieved_service_level"])

        return {
            "company_id": company_id,
            "timestamp": datetime.utcnow(),
            "status": "ok",
            "assumptions": {
                "service_level_definition": "Cycle Service Level (probability no stockout during lead time under ROP policy)",
                "demand_model": "Normal(mu_d, sigma_d) per day, aggregated over lead time",
                "lead_time_model": "Normal(LT, sigma_LT) days (sigma_LT may be 0)",
                "holding_cost_rate_annual_default": self.default_costs["holding_cost_rate_annual"],
                "stockout_penalty_default": self.default_costs["stockout_penalty_per_unit"],
                "simulation_runs": self.simulation["runs"] if validate_with_simulation else 0,
            },
            "scenarios_evaluated": scenario_results,
            "recommended_policy": recommended,
            "runtime_ms": int((datetime.utcnow() - t0).total_seconds() * 1000),
        }

    async def _evaluate_scenario(
        self,
        items: List[Dict[str, Any]],
        target_service_level: float,
        multiplier: float,
        validate_with_simulation: bool,
    ) -> Dict[str, Any]:
        """
        Compute SS/ROP and cost for each item, then aggregate.
        If validate_with_simulation: run Monte Carlo to estimate achieved service level.
        """
        z = _z_from_service_level(target_service_level)

        per_item = []
        total_holding_cost = 0.0
        total_stockout_cost = 0.0

        # For achieved CSL (simulate or estimate)
        achieved_csl_list = []

        # seed once per scenario for reproducibility
        rng = np.random.default_rng(self.simulation["random_seed"] + int(target_service_level * 10000) + int(multiplier * 100))

        for it in items:
            dem: ItemDemandStats = it["demand"]
            lt: LeadTimeStats = it["lead_time"]

            # Variance of demand during lead time:
            # Var(D_LT) = LT*sigma_d^2 + (mu_d^2)*sigma_LT^2
            var_dlt = lt.LT * (dem.sigma_d ** 2) + (dem.mu_d ** 2) * (lt.sigma_LT ** 2)
            sigma_dlt = float(np.sqrt(max(0.0, var_dlt)))
            mu_dlt = float(dem.mu_d * lt.LT)

            base_ss = z * sigma_dlt
            safety_stock = max(0.0, float(multiplier) * base_ss)
            rop = mu_dlt + safety_stock

            # Costs
            unit_cost = float(it["unit_cost"])
            h_rate = float(it["holding_cost_rate_annual"])
            days_per_year = float(self.default_costs["days_per_year"])
            holding_cost_per_unit_per_day = (unit_cost * h_rate) / days_per_year if unit_cost > 0 else 0.0

            # Approx expected on-hand buffer ~ safety_stock (simplification for policy comparison)
            # (You can refine with more inventory theory if you have order cycles.)
            expected_holding_cost = safety_stock * holding_cost_per_unit_per_day * days_per_year  # annualized

            stockout_penalty = float(it["stockout_penalty_per_unit"])

            if validate_with_simulation:
                achieved_csl, expected_short = self._simulate_csl_and_shortage(
                    rng=rng,
                    mu_d=dem.mu_d,
                    sigma_d=dem.sigma_d,
                    mu_lt=lt.LT,
                    sigma_lt=lt.sigma_LT,
                    rop=rop,
                    runs=self.simulation["runs"],
                )
            else:
                # No simulation: approximate CSL with Normal D_LT and ROP threshold
                # CSL ~= P(D_LT <= ROP) where D_LT ~ N(mu_dlt, sigma_dlt)
                achieved_csl = self._approx_csl(mu_dlt, sigma_dlt, rop)
                expected_short = self._approx_expected_shortage(mu_dlt, sigma_dlt, rop)

            expected_stockout_cost = expected_short * stockout_penalty  # per lead-time "event" unit shortage
            # Convert to annual-ish: assume reorder events per year ~ (annual demand / order qty).
            # If you don't have order qty, we keep it as "relative" and add as cost proxy:
            # For planning trade-offs, this is still useful. You can plug your order frequency later.
            expected_stockout_cost_annual_proxy = expected_stockout_cost * 12.0  # proxy: 12 cycles/year

            total_holding_cost += expected_holding_cost
            total_stockout_cost += expected_stockout_cost_annual_proxy
            achieved_csl_list.append(achieved_csl)

            per_item.append({
                "item_id": it["item_id"],
                "mu_d_per_day": dem.mu_d,
                "sigma_d_per_day": dem.sigma_d,
                "LT_days": lt.LT,
                "sigma_LT_days": lt.sigma_LT,
                "target_service_level": it["service_level_target"] or target_service_level,
                "z": z,
                "safety_stock_units": safety_stock,
                "reorder_point_units": rop,
                "holding_cost_annual_est": expected_holding_cost,
                "stockout_cost_annual_proxy": expected_stockout_cost_annual_proxy,
                "achieved_csl_est": achieved_csl,
            })

        # Aggregate achieved CSL: conservative = min across items or weighted average.
        # In practice, supply chain service often judged by worst critical SKUs.
        achieved_service = float(np.mean(achieved_csl_list)) if achieved_csl_list else 0.0

        total_expected_cost = total_holding_cost + total_stockout_cost

        return {
            "scenario": {
                "target_service_level": float(target_service_level),
                "safety_stock_multiplier": float(multiplier),
            },
            "achieved_service_level": achieved_service,
            "total_holding_cost_annual_est": total_holding_cost,
            "total_stockout_cost_annual_proxy": total_stockout_cost,
            "total_expected_cost": total_expected_cost,
            "items": per_item,
        }

    def _simulate_csl_and_shortage(
        self,
        rng: np.random.Generator,
        mu_d: float,
        sigma_d: float,
        mu_lt: float,
        sigma_lt: float,
        rop: float,
        runs: int,
    ) -> Tuple[float, float]:
        """
        Monte Carlo: sample lead time and demand during lead time, check if demand <= ROP.
        Also estimate expected shortage E[(D_LT - ROP)+].
        """
        # Sample lead time (truncate at small positive to avoid negative)
        if sigma_lt > 0:
            lt_samples = rng.normal(mu_lt, sigma_lt, size=runs)
            lt_samples = np.clip(lt_samples, 0.1, None)
        else:
            lt_samples = np.full(runs, max(0.1, mu_lt))

        # Demand during lead time: Normal with mean mu_d * LT, std sigma_d * sqrt(LT) (if sigma_d>0)
        # This assumes daily demands independent.
        mu_dlt = mu_d * lt_samples
        sigma_dlt = sigma_d * np.sqrt(lt_samples) if sigma_d > 0 else np.zeros_like(lt_samples)

        dlt_samples = rng.normal(mu_dlt, sigma_dlt)
        dlt_samples = np.clip(dlt_samples, 0.0, None)

        no_stockout = dlt_samples <= rop
        csl = float(np.mean(no_stockout))

        shortage = np.maximum(0.0, dlt_samples - rop)
        expected_shortage = float(np.mean(shortage))

        return csl, expected_shortage

    def _approx_csl(self, mu: float, sigma: float, rop: float) -> float:
        if sigma <= 1e-9:
            return 1.0 if mu <= rop else 0.0
        # Use error function approximation for normal CDF
        z = (rop - mu) / (sigma * np.sqrt(2.0))
        return float(0.5 * (1.0 + np.math.erf(z)))

    def _approx_expected_shortage(self, mu: float, sigma: float, rop: float) -> float:
        """
        Expected shortage for Normal demand: E[(X - r)+] = sigma * L(k) where k=(r-mu)/sigma, L(k)=phi(k)-k*(1-Phi(k))
        """
        if sigma <= 1e-9:
            return max(0.0, mu - rop)
        k = (rop - mu) / sigma
        # phi(k)
        phi = float(np.exp(-0.5 * k * k) / np.sqrt(2.0 * np.pi))
        # Phi(k)
        Phi = float(0.5 * (1.0 + np.math.erf(k / np.sqrt(2.0))))
        return float(sigma * (phi - k * (1.0 - Phi)))
    
    def _z_from_service_level(self, service_level: float) -> float:
        # common approximations
        table = {
            0.90: 1.282, 0.95: 1.645, 0.975: 1.960, 0.99: 2.326
        }
        closest = min(table.keys(), key=lambda k: abs(k - service_level))
        return table[closest]

    async def model_buffer_policy(
        self,
        company: Compare,
        mean_daily_demand: float,
        std_daily_demand: float,
        mean_lead_time_days: float,
        std_lead_time_days: float,
        unit_holding_cost_per_year: float,
        unit_stockout_cost: float,
    ) -> Dict[str, Any]:
        # Z-score from service level
        z = self._z_from_service_level(company.service_level_target)

        # Demand during lead time mean and std
        mu_lt = mean_daily_demand * mean_lead_time_days
        sigma_lt = math.sqrt(
            (mean_lead_time_days * (std_daily_demand ** 2)) +
            ((mean_daily_demand ** 2) * (std_lead_time_days ** 2))
        )

        safety_stock = z * sigma_lt
        reorder_point = mu_lt + safety_stock

        # Simple cost comparison
        holding_cost = safety_stock * (unit_holding_cost_per_year / 365.0) * mean_lead_time_days
        expected_stockout_units = max(0.0, (0.05 * sigma_lt))  # heuristic
        stockout_cost = expected_stockout_units * unit_stockout_cost

        return {
            "inputs": {
                "service_level_target": company.service_level_target,
                "mean_daily_demand": mean_daily_demand,
                "std_daily_demand": std_daily_demand,
                "mean_lead_time_days": mean_lead_time_days,
                "std_lead_time_days": std_lead_time_days,
                "unit_holding_cost_per_year": unit_holding_cost_per_year,
                "unit_stockout_cost": unit_stockout_cost
            },
            "outputs": {
                "z_value_used": z,
                "safety_stock_units": float(safety_stock),
                "reorder_point_units": float(reorder_point),
                "lead_time_demand_mean": float(mu_lt),
                "lead_time_demand_std": float(sigma_lt)
            },
            "cost_tradeoff": {
                "estimated_holding_cost_over_lead_time": float(holding_cost),
                "estimated_stockout_cost": float(stockout_cost),
                "recommended": "increase_buffer" if stockout_cost > holding_cost else "optimize_buffer"
            }
        }



# ----------------------------
# Example: how you call it
# ----------------------------
# async def demo(company):
#     modeler = BufferStockModeler()
#     items = [
#         {"item_id": "MCU", "mu_d": 300, "sigma_d": 80, "LT": 14, "sigma_LT": 3, "unit_cost": 12.0},
#         {"item_id": "SENSOR", "mu_d": 120, "sigma_d": 30, "LT": 21, "sigma_LT": 5, "unit_cost": 8.0},
#     ]
#     out = await modeler.model_buffer_stock_strategy(company, items, global_service_level_targets=[0.90, 0.95, 0.98])
#     print(out)