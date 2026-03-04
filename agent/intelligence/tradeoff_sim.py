from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
import math
import numpy as np
import logging

from agent.models import Company, MitigationAction

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _clamp01(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(v):
        return default
    return max(0.0, min(1.0, v))


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(v):
        return default
    return v


def _safe_corr(a: List[float], b: List[float]) -> float:
    """Returns correlation in [-1,1]. If undefined (zero variance), returns 0."""
    if len(a) < 2 or len(b) < 2:
        return 0.0
    x = np.array(a, dtype=float)
    y = np.array(b, dtype=float)
    # remove NaNs/inf
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return 0.0
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _dominates(a: Dict[str, float], b: Dict[str, float]) -> bool:
    """
    Multi-objective dominance:
    a dominates b if it's >= in all objectives and > in at least one.
    (Assumes higher is better for all objective scores.)
    """
    keys = a.keys()
    ge_all = all(a[k] >= b[k] for k in keys)
    gt_any = any(a[k] > b[k] for k in keys)
    return ge_all and gt_any


class TradeOffAnalyzer:
    def __init__(self) -> None:
        # Higher is better for ALL objective scores.
        self.objective_weights = {
            "cost": 0.30,          # cost score is “cheapness”
            "service_level": 0.30,
            "resilience": 0.25,
            "flexibility": 0.10,
            "roi": 0.05,           # small but realistic
        }

        self.scenario_types = [
            "cost_optimization",
            "service_level_maximization",
            "resilience_building",
            "balanced_approach",
        ]

        # Normalization constants (should eventually come from real data distribution)
        self.cost_scale = 1_000_000.0
        self.benefit_scale = 500_000.0

    async def analyze_mitigation_tradeoffs(
        self,
        company: Company,
        mitigation_actions: List[MitigationAction],
        *,
        budget_cap: Optional[float] = None,     # real-world constraint
        max_actions_per_scenario: int = 3
    ) -> Dict[str, Any]:
        logger.info("Analyzing trade-offs for %d mitigation actions", len(mitigation_actions))

        action_scores = [self._score_mitigation_action(company, a) for a in mitigation_actions]

        scenarios = self._generate_scenarios(
            action_scores,
            budget_cap=budget_cap,
            max_actions=max_actions_per_scenario,
        )

        tradeoff_analysis = self._analyze_tradeoffs(action_scores, scenarios)
        recommendations = self._generate_tradeoff_recommendations(company, tradeoff_analysis)

        return {
            "company_id": getattr(company, "id", None),
            "analysis_timestamp": _utc_now().isoformat(),
            "budget_cap": budget_cap,
            "mitigation_actions": action_scores,
            "scenarios": scenarios,
            "tradeoff_analysis": tradeoff_analysis,
            "recommendations": recommendations,
            "pareto_frontier": self._identify_pareto_frontier(action_scores),
        }

    # -------------------------
    # Scoring
    # -------------------------
    def _score_mitigation_action(self, company: Company, action: MitigationAction) -> Dict[str, Any]:
        """
        Produces objective scores in [0,1].
        Higher is better for all objectives (including 'cost' meaning "cheapness").
        """

        # raw inputs with safe defaults
        cost = max(0.0, _safe_float(getattr(action, "estimated_cost", 0.0), 0.0))
        benefit = max(0.0, _safe_float(getattr(action, "estimated_benefit", 0.0), 0.0))
        roi = _clamp01(getattr(action, "roi_score", 0.0), default=0.0)
        urgency = _clamp01(getattr(action, "urgency_score", 0.0), default=0.0)

        # normalize to scores
        # cost_score: cheapness -> 1 is cheap, 0 is expensive
        cost_score = 1.0 - min(1.0, cost / self.cost_scale)
        cost_score = _clamp01(cost_score)

        # benefit score (optional signal you can use later)
        benefit_score = min(1.0, benefit / self.benefit_scale)
        benefit_score = _clamp01(benefit_score)

        objectives = {
            "cost": cost_score,
            "service_level": self._service_level_impact(action),
            "resilience": self._resilience_impact(action),
            "flexibility": self._flexibility_impact(action),
            "roi": roi,
        }

        overall_score = float(sum(objectives[k] * w for k, w in self.objective_weights.items()))
        overall_score = _clamp01(overall_score)

        # risk-adjust: urgent actions may be less “choice-like” (or higher execution risk)
        risk_adjusted_score = overall_score * (1.0 - urgency * 0.2)
        risk_adjusted_score = _clamp01(risk_adjusted_score)

        return {
            "action_id": getattr(action, "id", None),
            "action_type": getattr(action, "action_type", "unknown"),
            "title": getattr(action, "title", "Untitled Action"),
            "objectives": objectives,
            "overall_score": overall_score,
            "risk_adjusted_score": risk_adjusted_score,
            "cost": cost,
            "benefit": benefit,
            "benefit_score": benefit_score,
            "roi": roi,
            "implementation_time_days": int(_safe_float(getattr(action, "implementation_time_days", 0), 0)),
            "priority": getattr(action, "priority_level", "medium"),
            "urgency_score": urgency,
        }

    def _service_level_impact(self, action: MitigationAction) -> float:
        base = {
            "rerouting": 0.7,
            "resourcing": 0.8,
            "buffering": 0.9,
            "negotiation": 0.6,
            "escalation": 0.5,
        }.get(getattr(action, "action_type", "unknown"), 0.5)

        mult = {
            "critical": 1.2,
            "high": 1.1,
            "medium": 1.0,
            "low": 0.9,
        }.get(getattr(action, "priority_level", "medium"), 1.0)

        return _clamp01(base * mult, default=0.5)

    def _resilience_impact(self, action: MitigationAction) -> float:
        base = {
            "rerouting": 0.6,
            "resourcing": 0.9,
            "buffering": 0.8,
            "negotiation": 0.5,
            "escalation": 0.4,
        }.get(getattr(action, "action_type", "unknown"), 0.5)

        t = max(0.0, _safe_float(getattr(action, "implementation_time_days", 0.0), 0.0))
        time_adjust = min(1.2, 1.0 + (t / 100.0))

        return _clamp01(base * time_adjust, default=0.5)

    def _flexibility_impact(self, action: MitigationAction) -> float:
        base = {
            "rerouting": 0.8,
            "resourcing": 0.7,
            "buffering": 0.4,
            "negotiation": 0.6,
            "escalation": 0.3,
        }.get(getattr(action, "action_type", "unknown"), 0.5)

        urgency = _clamp01(getattr(action, "urgency_score", 0.0), default=0.0)
        return _clamp01(base - urgency * 0.2, default=0.5)

    # -------------------------
    # Scenarios (multi-variable trade-off)
    # -------------------------
    def _generate_scenarios(
        self,
        action_scores: List[Dict[str, Any]],
        *,
        budget_cap: Optional[float],
        max_actions: int
    ) -> List[Dict[str, Any]]:
        scenarios: List[Dict[str, Any]] = []

        for scenario_type in self.scenario_types:
            scenario = self._create_scenario(action_scores, scenario_type, budget_cap, max_actions)
            scenarios.append(scenario)

        return scenarios

    def _create_scenario(
        self,
        action_scores: List[Dict[str, Any]],
        scenario_type: str,
        budget_cap: Optional[float],
        max_actions: int
    ) -> Dict[str, Any]:
        if not action_scores:
            return {
                "scenario_type": scenario_type,
                "selected_actions": [],
                "total_cost": 0.0,
                "total_benefit": 0.0,
                "roi": 0.0,
                "implementation_time_days": 0,
                "objectives": {},
                "overall_score": 0.0,
                "note": "No actions available",
            }

        # Rank actions based on scenario focus
        if scenario_type == "cost_optimization":
            ranked = sorted(action_scores, key=lambda x: x["cost"])  # cheaper first
        elif scenario_type == "service_level_maximization":
            ranked = sorted(action_scores, key=lambda x: x["objectives"]["service_level"], reverse=True)
        elif scenario_type == "resilience_building":
            ranked = sorted(action_scores, key=lambda x: x["objectives"]["resilience"], reverse=True)
        else:
            ranked = sorted(action_scores, key=lambda x: x["risk_adjusted_score"], reverse=True)

        # Select up to max_actions, respecting budget cap if provided
        selected: List[Dict[str, Any]] = []
        running_cost = 0.0

        for a in ranked:
            if len(selected) >= max_actions:
                break
            if budget_cap is not None and (running_cost + a["cost"]) > float(budget_cap):
                continue
            selected.append(a)
            running_cost += a["cost"]

        if not selected:
            return {
                "scenario_type": scenario_type,
                "selected_actions": [],
                "total_cost": 0.0,
                "total_benefit": 0.0,
                "roi": 0.0,
                "implementation_time_days": 0,
                "objectives": {},
                "overall_score": 0.0,
                "note": "No actions fit constraints (budget/max_actions).",
            }

        total_cost = float(sum(a["cost"] for a in selected))
        total_benefit = float(sum(a["benefit"] for a in selected))
        total_time = int(max(a["implementation_time_days"] for a in selected))

        scenario_objectives = {k: float(np.mean([a["objectives"][k] for a in selected])) for k in selected[0]["objectives"].keys()}
        overall = float(sum(scenario_objectives[k] * self.objective_weights[k] for k in self.objective_weights.keys()))

        return {
            "scenario_type": scenario_type,
            "selected_actions": [a["action_id"] for a in selected],
            "total_cost": total_cost,
            "total_benefit": total_benefit,
            "roi": (total_benefit / total_cost) if total_cost > 0 else 0.0,
            "implementation_time_days": total_time,
            "objectives": scenario_objectives,
            "overall_score": _clamp01(overall),
        }

    # -------------------------
    # Trade-off analytics
    # -------------------------
    def _analyze_tradeoffs(self, action_scores: List[Dict[str, Any]], scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not action_scores:
            return {
                "correlations": {},
                "best_options": {},
                "scenario_comparison": [],
                "tradeoff_insights": ["No actions available for trade-off analysis."],
            }

        cost = [a["objectives"]["cost"] for a in action_scores]
        service = [a["objectives"]["service_level"] for a in action_scores]
        resil = [a["objectives"]["resilience"] for a in action_scores]
        flex = [a["objectives"]["flexibility"] for a in action_scores]

        correlations = {
            "cost_service": _safe_corr(cost, service),
            "cost_resilience": _safe_corr(cost, resil),
            "service_resilience": _safe_corr(service, resil),
            "resilience_flexibility": _safe_corr(resil, flex),
        }

        best_options = {
            "cost": max(action_scores, key=lambda x: x["objectives"]["cost"]),
            "service_level": max(action_scores, key=lambda x: x["objectives"]["service_level"]),
            "resilience": max(action_scores, key=lambda x: x["objectives"]["resilience"]),
            "flexibility": max(action_scores, key=lambda x: x["objectives"]["flexibility"]),
            "risk_adjusted": max(action_scores, key=lambda x: x["risk_adjusted_score"]),
        }

        scenario_comparison = [
            {
                "type": s["scenario_type"],
                "cost": s["total_cost"],
                "benefit": s["total_benefit"],
                "roi": s["roi"],
                "implementation_time": s["implementation_time_days"],
                "objectives": s.get("objectives", {}),
                "overall_score": s.get("overall_score", 0.0),
            }
            for s in scenarios
        ]

        insights = self._generate_tradeoff_insights(correlations, best_options)
        return {
            "correlations": correlations,
            "best_options": best_options,
            "scenario_comparison": scenario_comparison,
            "tradeoff_insights": insights,
        }

    def _generate_tradeoff_insights(self, correlations: Dict[str, float], best_options: Dict[str, Any]) -> List[str]:
        insights: List[str] = []

        if correlations.get("cost_service", 0.0) < -0.5:
            insights.append("Strong trade-off: cheaper options tend to reduce service level.")
        if correlations.get("cost_resilience", 0.0) < -0.5:
            insights.append("Resilience tends to require investment (cheapness vs resilience trade-off).")
        if correlations.get("service_resilience", 0.0) > 0.5:
            insights.append("Synergy: improving service level often improves resilience too.")

        cost_best = best_options["cost"]
        svc_best = best_options["service_level"]
        if cost_best["action_id"] != svc_best["action_id"]:
            insights.append(f"Cost-optimal ({cost_best['title']}) differs from service-optimal ({svc_best['title']}).")

        ra_best = best_options["risk_adjusted"]
        insights.append(f"Best overall (risk-adjusted) action: {ra_best['title']}.")

        return insights

    def _generate_tradeoff_recommendations(self, company: Company, tradeoff_analysis: Dict[str, Any]) -> List[str]:
        recs: List[str] = []

        risk_appetite = _clamp01(getattr(company, "risk_appetite", 0.5), default=0.5)

        if risk_appetite < 0.3:
            recs.append("Risk-averse profile: prioritize resilience and service-level even if cost increases.")
        elif risk_appetite > 0.7:
            recs.append("Risk-tolerant profile: prioritize cost/ROI, keep flexibility for reactive response.")
        else:
            recs.append("Balanced profile: pick risk-adjusted best action plus one resilience booster.")

        scenarios = tradeoff_analysis.get("scenario_comparison", [])
        if scenarios:
            best_roi = max(scenarios, key=lambda x: x.get("roi", 0.0))
            recs.append(f"Highest-ROI scenario: {best_roi['type']} (ROI={best_roi['roi']:.2f}).")

        return recs

    def _identify_pareto_frontier(self, action_scores: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        True multi-objective Pareto frontier across:
        cost, service_level, resilience, flexibility, roi
        (all in [0,1], higher is better)
        """
        if len(action_scores) < 2:
            return action_scores

        points = []
        for a in action_scores:
            points.append({
                "action_id": a["action_id"],
                "title": a["title"],
                "objectives": a["objectives"],  # includes roi
                "overall_score": a["overall_score"],
                "risk_adjusted_score": a["risk_adjusted_score"],
                "cost_raw": a["cost"],
                "benefit_raw": a["benefit"],
            })

        pareto = []
        for i, p in enumerate(points):
            dominated = False
            for j, q in enumerate(points):
                if i == j:
                    continue
                if _dominates(q["objectives"], p["objectives"]):
                    dominated = True
                    break
            if not dominated:
                pareto.append(p)

        return pareto