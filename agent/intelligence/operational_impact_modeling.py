from __future__ import annotations

from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import logging

from agent.models import Company, Supplier, Disruption

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_utc_now() -> str:
    return _utc_now().isoformat()


def _clamp01(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    if v != v:  # NaN
        return default
    return max(0.0, min(1.0, v))


def _safe_str(x: Any, default: str = "") -> str:
    try:
        s = str(x)
        return s if s else default
    except Exception:
        return default


class ImpactModeler:
    """
    Operational Impact Modeler (safe, deterministic, no external APIs).

    - Uses clamped 0..1 inputs.
    - Avoids timezone-naive datetimes (returns ISO-UTC).
    - Does NOT assume Company/Disruption fields always exist (uses getattr defaults).
    - simulate_scenario_impacts tries multiple Disruption constructors to avoid crashes.
    """

    def __init__(self) -> None:
        # Weighting across impact dimensions
        self.impact_weights: Dict[str, float] = {
            "revenue_impact": 0.35,
            "operational_impact": 0.25,
            "customer_impact": 0.20,
            "cost_impact": 0.20,
        }

        # Industry multipliers (optional)
        self.industry_multipliers: Dict[str, Dict[str, float]] = {
            "automotive": {"revenue": 1.2, "operational": 1.3, "customer": 1.1, "cost": 1.2},
            "electronics": {"revenue": 1.4, "operational": 1.2, "customer": 1.3, "cost": 1.1},
            "manufacturing": {"revenue": 1.1, "operational": 1.4, "customer": 1.0, "cost": 1.3},
        }

        # Disruption-type timing assumptions (days)
        self.disruption_time_factors: Dict[str, int] = {
            "shipping": 14,
            "geopolitical": 30,
            "supplier": 3,
            "climate": 1,
            "custom": 7,
        }

        # Disruption-type baseline durations (days)
        self.base_durations: Dict[str, int] = {
            "shipping": 30,
            "geopolitical": 90,
            "supplier": 45,
            "climate": 14,
            "custom": 30,
        }

    async def model_disruption_impact(
        self, company: Company, disruption: Disruption, suppliers: List[Supplier]
    ) -> Dict[str, Any]:
        """Model the comprehensive impact of a disruption on the company."""
        disruption_id = getattr(disruption, "id", None)
        company_id = getattr(company, "id", None)

        logger.info("Modeling impact of disruption=%s company=%s", disruption_id, company_id)

        revenue_impact = await self._calculate_revenue_impact(company, disruption, suppliers)
        operational_impact = await self._calculate_operational_impact(company, disruption, suppliers)
        customer_impact = await self._calculate_customer_impact(company, disruption, suppliers)
        cost_impact = await self._calculate_cost_impact(company, disruption, suppliers)

        overall = (
            revenue_impact["impact_score"] * self.impact_weights["revenue_impact"]
            + operational_impact["impact_score"] * self.impact_weights["operational_impact"]
            + customer_impact["impact_score"] * self.impact_weights["customer_impact"]
            + cost_impact["impact_score"] * self.impact_weights["cost_impact"]
        )
        overall = _clamp01(overall)

        revenue_at_risk = await self._estimate_revenue_at_risk(company, overall)
        time_to_impact = await self._estimate_time_to_impact(company, disruption)
        duration = await self._estimate_impact_duration(disruption)
        affected_products = await self._identify_affected_products(disruption)
        mitigation = await self._assess_mitigation_potential(company)

        return {
            "disruption_id": disruption_id,
            "company_id": company_id,
            "overall_impact_score": overall,
            "impact_level": self._determine_impact_level(overall),
            "revenue_impact": revenue_impact,
            "operational_impact": operational_impact,
            "customer_impact": customer_impact,
            "cost_impact": cost_impact,
            "revenue_at_risk": revenue_at_risk,
            "time_to_impact_days": time_to_impact,
            "impact_duration_days": duration,
            "affected_products": affected_products,
            "mitigation_potential": mitigation,
            "confidence_level": 0.8,
            "modeled_at": _iso_utc_now(),
        }

    # -------------------------
    # Dimension calculators
    # -------------------------
    async def _calculate_revenue_impact(
        self, company: Company, disruption: Disruption, suppliers: List[Supplier]
    ) -> Dict[str, Any]:
        base = _clamp01(getattr(disruption, "severity_score", 0.5), default=0.5)
        mult = self._industry_mult(company)

        conc = _clamp01(getattr(company, "supplier_concentration_risk", 0.0), default=0.0)
        concentration_factor = 1.0 + (conc * 0.5)

        impact_score = _clamp01(base * mult["revenue"] * concentration_factor)

        # Prefer real company revenue if available; otherwise fallback
        annual_revenue = float(getattr(company, "annual_revenue", 50_000_000.0))
        daily_revenue = annual_revenue / 365.0
        estimated_daily_loss = impact_score * daily_revenue

        return {
            "impact_score": impact_score,
            "estimated_daily_loss": estimated_daily_loss,
            "estimated_monthly_loss": estimated_daily_loss * 30.0,
            "recovery_timeline_days": int(impact_score * 90),
            "key_drivers": ["production delays", "order cancellations", "price discounts"],
        }

    async def _calculate_operational_impact(
        self, company: Company, disruption: Disruption, suppliers: List[Supplier]
    ) -> Dict[str, Any]:
        """
        Operational impact modeling (still lightweight, but non-crashy):
        uses severity + industry multiplier + lead-time sensitivity + inventory buffer policy.
        """
        base = _clamp01(getattr(disruption, "severity_score", 0.5), default=0.5)
        mult = self._industry_mult(company)

        lead_sens = _clamp01(getattr(company, "lead_time_sensitivity", 0.0), default=0.0)
        inv_buffer = _clamp01(getattr(company, "inventory_buffer_policy", 0.0), default=0.0)

        # Higher lead-time sensitivity increases operational impact; more buffer reduces it.
        sensitivity_adjustment = 1.0 + (lead_sens * 0.3)
        buffer_reduction = 1.0 - (inv_buffer * 0.25)

        impact_score = _clamp01(base * mult["operational"] * sensitivity_adjustment * buffer_reduction)

        # Map score to interpretable operational outputs
        production_delay_days = int(round(impact_score * 45))
        inventory_disruption_pct = impact_score * 100.0
        workforce_impact_pct = impact_score * 30.0

        return {
            "impact_score": impact_score,
            "production_delay_days": production_delay_days,
            "inventory_disruption_percentage": inventory_disruption_pct,
            "workforce_impact_percentage": workforce_impact_pct,
            "key_drivers": ["supply shortages", "production bottlenecks", "quality issues"],
        }

    async def _calculate_customer_impact(
        self, company: Company, disruption: Disruption, suppliers: List[Supplier]
    ) -> Dict[str, Any]:
        base = _clamp01(getattr(disruption, "severity_score", 0.5), default=0.5)
        mult = self._industry_mult(company)

        service_target = _clamp01(getattr(company, "service_level_target", 0.95), default=0.95)
        service_factor = 2.0 - service_target  # higher target => higher sensitivity

        impact_score = _clamp01(base * mult["customer"] * service_factor)

        return {
            "impact_score": impact_score,
            "delayed_orders_percentage": impact_score * 80.0,
            "customer_satisfaction_drop": impact_score * 50.0,
            "estimated_churn_rate": impact_score * 0.05,
            "key_drivers": ["delivery delays", "quality issues", "communication gaps"],
        }

    async def _calculate_cost_impact(
        self, company: Company, disruption: Disruption, suppliers: List[Supplier]
    ) -> Dict[str, Any]:
        base = _clamp01(getattr(disruption, "severity_score", 0.5), default=0.5)
        mult = self._industry_mult(company)

        impact_score = _clamp01(base * mult["cost"])

        return {
            "impact_score": impact_score,
            "expedited_shipping_increase": impact_score * 200.0,
            "premium_pricing_increase": impact_score * 150.0,
            "inventory_carrying_increase": impact_score * 50.0,
            "key_drivers": ["expedited shipping", "premium sourcing", "increased inventory"],
        }

    # -------------------------
    # Derived metrics
    # -------------------------
    async def _estimate_revenue_at_risk(self, company: Company, impact_score: float) -> Dict[str, Any]:
        annual_revenue = float(getattr(company, "annual_revenue", 50_000_000.0))
        impact_score = _clamp01(impact_score)

        # Up to 30% of revenue at risk (assumption)
        risk_percentage = impact_score * 0.3
        quarterly_rar = annual_revenue * risk_percentage / 4.0

        return {
            "annual_revenue": annual_revenue,
            "risk_percentage": risk_percentage,
            "quarterly_revenue_at_risk": quarterly_rar,
            "confidence_level": 0.75,
        }

    async def _estimate_time_to_impact(self, company: Company, disruption: Disruption) -> int:
        dtype = _safe_str(getattr(disruption, "disruption_type", "custom"), default="custom").lower()
        base_days = int(self.disruption_time_factors.get(dtype, self.disruption_time_factors["custom"]))

        lead_sens = _clamp01(getattr(company, "lead_time_sensitivity", 0.0), default=0.0)
        sensitivity_adjustment = 1.0 - (lead_sens * 0.3)

        return max(1, int(round(base_days * sensitivity_adjustment)))

    async def _estimate_impact_duration(self, disruption: Disruption) -> int:
        dtype = _safe_str(getattr(disruption, "disruption_type", "custom"), default="custom").lower()
        base_duration = int(self.base_durations.get(dtype, self.base_durations["custom"]))

        severity = _clamp01(getattr(disruption, "severity_score", 0.5), default=0.5)
        severity_multiplier = 0.5 + (severity * 1.5)
        return max(1, int(round(base_duration * severity_multiplier)))

    async def _identify_affected_products(self, disruption: Disruption) -> List[str]:
        # Safe mock list (replace with real product catalog later)
        products = [
            "Automotive Control Units",
            "Industrial Sensors",
            "Consumer Electronics Components",
            "Manufacturing Equipment Parts",
        ]
        severity = _clamp01(getattr(disruption, "severity_score", 0.5), default=0.5)
        n = max(1, int(len(products) * severity))
        return products[:n]

    async def _assess_mitigation_potential(self, company: Company) -> Dict[str, Any]:
        base_potential = 0.7

        conc = _clamp01(getattr(company, "supplier_concentration_risk", 0.0), default=0.0)
        inv_buf = _clamp01(getattr(company, "inventory_buffer_policy", 0.0), default=0.0)

        diversification_bonus = conc * -0.2  # lower concentration => better mitigation
        inventory_bonus = inv_buf * 0.2

        mitigation = max(0.2, min(1.0, base_potential + diversification_bonus + inventory_bonus))
        return {
            "mitigation_potential_score": mitigation,
            "expected_reduction_percentage": mitigation * 100.0,
            "key_mitigation_levers": ["supplier diversification", "inventory buffering", "alternative logistics"],
            "time_to_mitigate_days": int(round((1.0 - mitigation) * 60)),
        }

    # -------------------------
    # Scenario simulation
    # -------------------------
    async def simulate_scenario_impacts(self, company: Company, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a Disruption safely from a dict and run model_disruption_impact.

        This avoids constructor mismatch bugs by trying common Disruption signatures.
        """
        scenario_type = _safe_str(scenario.get("type", "custom"), default="custom").lower()
        params = scenario.get("parameters", {}) or {}

        disruption_id = scenario.get("id") or f"sim-{int(_utc_now().timestamp())}"
        title = scenario.get("title", "Custom Scenario")
        description = scenario.get("description", "Simulated disruption scenario")
        severity = params.get("severity", 0.5)
        regions = params.get("regions", [])

        mock_disruption = self._build_disruption_safely(
            disruption_id=disruption_id,
            title=title,
            description=description,
            disruption_type=scenario_type,
            severity_score=severity,
            affected_regions=regions,
        )

        impact_analysis = await self.model_disruption_impact(company, mock_disruption, [])

        impact_analysis["scenario_type"] = scenario_type
        impact_analysis["scenario_parameters"] = params
        impact_analysis["simulation_timestamp"] = _iso_utc_now()
        return impact_analysis

    def _build_disruption_safely(
        self,
        *,
        disruption_id: str,
        title: str,
        description: str,
        disruption_type: str,
        severity_score: Any,
        affected_regions: Any,
    ) -> Disruption:
        """
        Tries multiple Disruption constructors to avoid runtime crashes
        if your model has slightly different required fields.
        """
        severity_score_f = _clamp01(severity_score, default=0.5)

        # Most complete signature (matches your earlier code)
        try:
            return Disruption(
                id=disruption_id,
                title=title,
                description=description,
                disruption_type=disruption_type,
                severity_score=severity_score_f,
                affected_regions=affected_regions,
            )
        except TypeError:
            pass

        # Common simplified signatures
        try:
            return Disruption(
                id=disruption_id,
                disruption_type=disruption_type,
                severity_score=severity_score_f,
            )
        except TypeError:
            pass

        try:
            return Disruption(
                disruption_type=disruption_type,
                severity_score=severity_score_f,
            )
        except TypeError as e:
            # At this point, your Disruption model is very different.
            # Raise a clear error instead of failing later with attribute errors.
            raise TypeError(
                "Disruption constructor doesn't match expected fields. "
                "Please paste src.models.Disruption definition so I can align it."
            ) from e

    # -------------------------
    # Utilities
    # -------------------------
    def _industry_mult(self, company: Company) -> Dict[str, float]:
        industry = _safe_str(getattr(company, "industry", ""), default="").lower()
        return self.industry_multipliers.get(
            industry, {"revenue": 1.0, "operational": 1.0, "customer": 1.0, "cost": 1.0}
        )

    def _determine_impact_level(self, impact_score: float) -> str:
        score = _clamp01(impact_score)
        if score >= 0.8:
            return "severe"
        if score >= 0.6:
            return "high"
        if score >= 0.4:
            return "moderate"
        return "low"