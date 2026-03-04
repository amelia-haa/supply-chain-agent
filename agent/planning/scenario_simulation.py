from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
import math
import random
import numpy as np
import logging
from agent.models.supply_chain import Company, Supplier

logger = logging.getLogger(__name__)


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clamp01(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(v):
        return default
    return max(0.0, min(1.0, v))


def _safe_float(x: Any, default: float) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(v):
        return default
    return v


@dataclass(frozen=True)
class LeverConfig:
    buffer_days: float               # extra buffer coverage in days
    expedite_fill: float             # fraction of unmet demand recovered via expedite (0..1)
    dual_source_fraction: float      # fraction of demand protected by dual sourcing (0..1)
    reroute_capacity: float          # fraction of disruption impact reduced by reroute (0..1)


class ScenarioSimulator:
    """
    Scenario Simulation for cost vs service trade-offs.

    Core idea:
    - Disruptions cause supply shortfall (random severity + duration).
    - Buffers cover early days of shortfall.
    - Dual sourcing reduces shortfall probability/impact.
    - Rerouting reduces impact for shipping-like disruptions.
    - Expediting recovers some unmet demand but costs money.

    Outputs:
    - service_level (fill rate) over horizon
    - total_cost (holding + expedite + switching/reroute)
    - revenue_at_risk estimate (optional proxy, based on demand value)
    - risk metrics: VaR/CVaR of cost, service shortfall
    """

    def __init__(self, *, monte_carlo_runs: int = 2000, seed: int = 42) -> None:
        self.monte_carlo_runs = int(monte_carlo_runs)
        self.seed = seed

        # Simple disruption priors (you can replace with learned values from your risk engine)
        self.disruption_types = {
            "shipping": {"p": 0.12, "sev_range": (0.3, 0.9), "dur_range": (7, 90)},
            "supplier": {"p": 0.18, "sev_range": (0.4, 0.95), "dur_range": (14, 180)},
            "geopolitical": {"p": 0.06, "sev_range": (0.4, 0.9), "dur_range": (30, 365)},
            "climate": {"p": 0.10, "sev_range": (0.3, 0.9), "dur_range": (3, 45)},
        }
    
    async def simulate_cost_vs_service(self, company: Company, suppliers: List[Supplier]) -> Dict[str, Any]:
        """
        Output: distributions of cost and service impacts under random disruptions.
        """
        runs = []
        for _ in range(self.monte_carlo_runs):
            # random disruption severity
            sev = random.uniform(0.0, 1.0)
            leadtime_shock = random.randint(0, 25)
            cost_shock = random.uniform(0.0, 0.25)

            # service drops with severity + leadtime shock and supplier risk
            avg_supplier_risk = np.mean([s.risk_score for s in suppliers]) if suppliers else 0.3
            service_level = max(0.0, company.service_level_target - (0.2 * sev) - (leadtime_shock/200) - (0.1*avg_supplier_risk))

            # cost increases with cost shock + expediting when service low
            expediting = max(0.0, (company.service_level_target - service_level)) * 0.4
            total_cost_multiplier = 1.0 + cost_shock + expediting

            runs.append({
                "severity": sev,
                "service_level": service_level,
                "cost_multiplier": total_cost_multiplier
            })

        service = [r["service_level"] for r in runs]
        costm = [r["cost_multiplier"] for r in runs]

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "runs": self.monte_carlo_runs,
            "service": {
                "mean": float(np.mean(service)),
                "p05": float(np.percentile(service, 5)),
                "p50": float(np.percentile(service, 50)),
                "p95": float(np.percentile(service, 95)),
            },
            "cost_multiplier": {
                "mean": float(np.mean(costm)),
                "p05": float(np.percentile(costm, 5)),
                "p50": float(np.percentile(costm, 50)),
                "p95": float(np.percentile(costm, 95)),
            }
        }

    async def run_tradeoff_simulation(
        self,
        company: Company,
        *,
        horizon_days: int = 90,
        # demand_value_per_day is total “value of demand” per day (revenue proxy)
        demand_value_per_day: Optional[float] = None,
        # cost parameters (replace with real finance/ERP later)
        holding_cost_per_day_per_buffer_day: float = 1500.0,
        expedite_cost_rate: float = 0.12,      # % of recovered demand value spent as expedite cost
        dual_source_cost_per_day: float = 3000.0,
        reroute_cost_per_day: float = 2000.0,
        # grid of configs to test
        configs: Optional[List[LeverConfig]] = None,
    ) -> Dict[str, Any]:
        random.seed(self.seed)
        np.random.seed(self.seed)

        H = int(horizon_days)
        H = 1 if H < 1 else (3650 if H > 3650 else H)

        # revenue proxy: try company.annual_revenue; else fallback
        annual_rev = _safe_float(getattr(company, "annual_revenue", 50_000_000.0), 50_000_000.0)
        if demand_value_per_day is None:
            demand_value_per_day = max(0.0, annual_rev / 365.0)

        demand_value_per_day = max(0.0, float(demand_value_per_day))

        # default config grid (cost vs service trade-off curve)
        if configs is None:
            configs = [
                LeverConfig(buffer_days=0,  expedite_fill=0.00, dual_source_fraction=0.00, reroute_capacity=0.00),
                LeverConfig(buffer_days=7,  expedite_fill=0.10, dual_source_fraction=0.10, reroute_capacity=0.10),
                LeverConfig(buffer_days=14, expedite_fill=0.20, dual_source_fraction=0.20, reroute_capacity=0.20),
                LeverConfig(buffer_days=21, expedite_fill=0.30, dual_source_fraction=0.30, reroute_capacity=0.30),
                LeverConfig(buffer_days=30, expedite_fill=0.40, dual_source_fraction=0.40, reroute_capacity=0.40),
            ]

        results = []
        for cfg in configs:
            metrics = self._simulate_config(
                H=H,
                demand_value_per_day=demand_value_per_day,
                cfg=cfg,
                holding_cost_per_day_per_buffer_day=holding_cost_per_day_per_buffer_day,
                expedite_cost_rate=expedite_cost_rate,
                dual_source_cost_per_day=dual_source_cost_per_day,
                reroute_cost_per_day=reroute_cost_per_day,
            )
            results.append(metrics)

        # Identify “best” according to different preferences
        best_cost = min(results, key=lambda r: r["cost_mean"])
        best_service = max(results, key=lambda r: r["service_level_mean"])
        best_balanced = max(results, key=lambda r: r["utility_score"])

        # Pareto frontier (min cost, max service)
        pareto = self._pareto_frontier_cost_service(results)

        return {
            "company_id": getattr(company, "id", None),
            "timestamp_utc": _utc_iso(),
            "horizon_days": H,
            "demand_value_per_day": demand_value_per_day,
            "runs": self.monte_carlo_runs,
            "configs_tested": [cfg.__dict__ for cfg in configs],
            "results": results,
            "best": {
                "min_cost": best_cost,
                "max_service": best_service,
                "balanced": best_balanced,
            },
            "pareto_frontier_cost_vs_service": pareto,
        }

    # -----------------------------
    # Core simulator
    # -----------------------------
    def _simulate_config(
        self,
        *,
        H: int,
        demand_value_per_day: float,
        cfg: LeverConfig,
        holding_cost_per_day_per_buffer_day: float,
        expedite_cost_rate: float,
        dual_source_cost_per_day: float,
        reroute_cost_per_day: float,
    ) -> Dict[str, Any]:
        buffer_days = max(0.0, float(cfg.buffer_days))
        expedite_fill = _clamp01(cfg.expedite_fill, 0.0)
        dual = _clamp01(cfg.dual_source_fraction, 0.0)
        reroute = _clamp01(cfg.reroute_capacity, 0.0)

        # store trial outcomes for distributions
        service_levels = []
        total_costs = []
        revenue_at_risk = []

        for _ in range(self.monte_carlo_runs):
            # simulate disruptions over horizon
            unmet_value = 0.0
            disrupted_days_total = 0

            for dtype, params in self.disruption_types.items():
                # dual sourcing reduces probability that disruption affects you (simple model)
                p_effective = params["p"] * (1.0 - 0.7 * dual)

                if random.random() < p_effective:
                    sev = random.uniform(*params["sev_range"])
                    dur = random.randint(*params["dur_range"])
                    dur = min(dur, H)

                    # rerouting reduces shipping disruption impact mostly; mild effect otherwise
                    if dtype == "shipping":
                        sev = sev * (1.0 - 0.8 * reroute)
                    else:
                        sev = sev * (1.0 - 0.2 * reroute)

                    sev = _clamp01(sev, 0.0)
                    disrupted_days_total += dur

                    # buffer covers first buffer_days of disruption duration
                    covered = min(dur, int(buffer_days))
                    exposed = max(0, dur - covered)

                    # exposed demand shortfall value
                    shortfall_value = exposed * demand_value_per_day * sev

                    # expedite recovers some shortfall (but costs money later)
                    recovered = shortfall_value * expedite_fill
                    remaining = shortfall_value - recovered

                    unmet_value += remaining

            # compute service level as fraction of demand met
            total_demand = demand_value_per_day * H
            met = max(0.0, total_demand - unmet_value)
            service_level = 1.0 if total_demand <= 0 else (met / total_demand)
            service_level = max(0.0, min(1.0, service_level))

            # cost model:
            # holding cost scales with buffer_days and horizon
            holding_cost = buffer_days * holding_cost_per_day_per_buffer_day * H / 30.0  # monthly-ish scaling
            # dual sourcing cost scales with horizon and dual fraction
            dual_cost = dual_source_cost_per_day * dual * H
            # reroute cost scales with horizon and reroute usage
            reroute_cost = reroute_cost_per_day * reroute * H
            # expedite cost: pay % of recovered demand value (we approximate by using unmet_value and recovery rate)
            # If unmet is remaining after expedite, recovered approx = unmet/(1-expedite_fill) - unmet
            recovered_est = 0.0
            if expedite_fill > 0 and expedite_fill < 1:
                recovered_est = (unmet_value / (1.0 - expedite_fill)) - unmet_value
            expedite_cost = recovered_est * expedite_cost_rate

            total_cost = holding_cost + dual_cost + reroute_cost + expedite_cost

            service_levels.append(service_level)
            total_costs.append(total_cost)
            revenue_at_risk.append(unmet_value)

        # summarize distributions
        sl = np.array(service_levels, dtype=float)
        tc = np.array(total_costs, dtype=float)
        rar = np.array(revenue_at_risk, dtype=float)

        # utility: balance cost vs service (you can tune weights)
        # Normalize cost by demand scale to avoid domination by big companies
        cost_norm = np.mean(tc) / (demand_value_per_day * H + 1e-9)
        utility = float(np.mean(sl) - 0.5 * cost_norm)  # higher better

        return {
            "config": cfg.__dict__,
            "service_level_mean": float(np.mean(sl)),
            "service_level_p10": float(np.percentile(sl, 10)),
            "service_level_p50": float(np.percentile(sl, 50)),
            "service_level_p90": float(np.percentile(sl, 90)),
            "cost_mean": float(np.mean(tc)),
            "cost_var_95": float(np.percentile(tc, 95)),
            "cost_cvar_95": float(np.mean(tc[tc >= np.percentile(tc, 95)])) if tc.size > 0 else 0.0,
            "revenue_at_risk_mean": float(np.mean(rar)),
            "revenue_at_risk_var_95": float(np.percentile(rar, 95)),
            "utility_score": utility,
        }

    def _pareto_frontier_cost_service(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Pareto frontier for 2D:
        - minimize cost_mean
        - maximize service_level_mean
        """
        pts = [(r["cost_mean"], r["service_level_mean"], r) for r in results]
        pts.sort(key=lambda x: x[0])  # sort by cost

        frontier = []
        best_service = -1.0
        for cost, service, r in pts:
            if service > best_service:
                frontier.append(r)
                best_service = service
        return frontier
