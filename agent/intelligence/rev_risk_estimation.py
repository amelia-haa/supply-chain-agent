from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone
import math
import random


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


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(v):
        return default
    return v


@dataclass(frozen=True)
class RevenueAtRiskResult:
    timestamp_utc: str
    horizon_days: int
    n_trials: int

    # Expected value outputs (probability-weighted)
    expected_revenue_at_risk: float
    expected_margin_at_risk: float

    # Conditional outputs (if disruption happens)
    revenue_at_risk_if_disrupted: float
    margin_at_risk_if_disrupted: float

    # Risk fractions
    disruption_probability: float
    fraction_of_revenue_at_risk_if_disrupted: float
    fraction_of_revenue_at_risk_expected: float

    # Diagnostics
    top_affected_products: List[Tuple[str, float]]
    notes: List[str]


class RiskIntelligenceEngine:
    # ... your other methods ...

    def estimate_revenue_at_risk_real(
        self,
        *,
        # ---- Core probability inputs ----
        disruption_probability: float,
        expected_duration_days: int,
        severity: float,

        # ---- Real business data ----
        annual_revenue: float,
        # sales_by_product_per_day: {product: revenue_per_day}
        sales_by_product_per_day: Dict[str, float],

        # optional: margin_by_product: {product: contribution_margin_fraction}, e.g. 0.25
        margin_by_product: Optional[Dict[str, float]] = None,

        # ---- Supply chain structure (optional but “real”) ----
        # bom: {product: {component: units_needed_per_unit_product}}
        # If you don't model units, you can still use it as "dependency map".
        bom: Optional[Dict[str, Dict[str, float]]] = None,

        # ---- Disruption mapping ----
        # affected_components: {component: supply_reduction_fraction} (0..1), 1 means fully blocked
        affected_components: Optional[Dict[str, float]] = None,

        # affected_products_direct: {product: supply_reduction_fraction} (0..1)
        # Use this if you don't have BOM yet.
        affected_products_direct: Optional[Dict[str, float]] = None,

        # ---- Horizon + simulation ----
        horizon_days: int = 90,
        n_trials: int = 2000,
        seed: int = 42,

        # ---- Simple mitigation knobs (realistic levers) ----
        # inventory_buffer_days: {product: days_of_sales_covered}
        inventory_buffer_days: Optional[Dict[str, float]] = None,

        # fraction_of_demand_you_can_expedite: 0..1 (expedite or alternate sourcing)
        expedite_fill_fraction: float = 0.0,

        # optional cost: expedite_cost_rate is fraction of revenue lost recovered via expedite cost
        expedite_cost_rate: float = 0.0,
    ) -> RevenueAtRiskResult:
        """
        Real-world-ish Revenue-at-Risk estimator.

        What it does:
        1) Determines which products are impacted (via BOM->components or direct product mapping).
        2) Converts severity into a supply shortfall factor.
        3) Simulates disruption duration uncertainty + partial fulfillment (Monte Carlo).
        4) Computes revenue-at-risk and margin-at-risk over the horizon.
        5) Returns both:
           - conditional RaR (if disruption happens)
           - expected RaR (probability-weighted)

        This is robust:
        - clamps inputs
        - handles missing margin/BOM gracefully
        - will not crash on missing keys
        """

        notes: List[str] = []

        # ---- sanitize inputs ----
        p = _clamp01(disruption_probability, default=0.0)
        sev = _clamp01(severity, default=0.0)
        H = int(horizon_days)
        H = 1 if H < 1 else (3650 if H > 3650 else H)

        dur = int(expected_duration_days)
        dur = 1 if dur < 1 else (3650 if dur > 3650 else dur)

        annual_rev = max(0.0, _safe_float(annual_revenue, 0.0))
        if annual_rev <= 0:
            notes.append("annual_revenue <= 0; revenue-at-risk will be 0 unless sales_by_product_per_day has values.")

        # sales inputs
        sales = {k: max(0.0, _safe_float(v, 0.0)) for k, v in (sales_by_product_per_day or {}).items()}
        if not sales:
            notes.append("sales_by_product_per_day is empty -> cannot compute meaningful RaR.")
            # return safe output
            return RevenueAtRiskResult(
                timestamp_utc=_utc_iso(),
                horizon_days=H,
                n_trials=0,
                expected_revenue_at_risk=0.0,
                expected_margin_at_risk=0.0,
                revenue_at_risk_if_disrupted=0.0,
                margin_at_risk_if_disrupted=0.0,
                disruption_probability=p,
                fraction_of_revenue_at_risk_if_disrupted=0.0,
                fraction_of_revenue_at_risk_expected=0.0,
                top_affected_products=[],
                notes=notes,
            )

        # margins
        margins = margin_by_product or {}
        # if missing for a product, assume a conservative default margin fraction
        default_margin = 0.25
        expedite_fill = _clamp01(expedite_fill_fraction, default=0.0)
        expedite_cost_rate = _clamp01(expedite_cost_rate, default=0.0)

        # inventory buffer
        buffer_days = inventory_buffer_days or {}

        # ---- determine product impact mapping ----
        # We want: product -> supply_reduction_fraction (0..1)
        product_reduction: Dict[str, float] = {}

        if affected_products_direct:
            for prod, red in affected_products_direct.items():
                product_reduction[prod] = _clamp01(red, default=0.0)

        # If BOM + affected components exist, compute product reduction as max reduction among required components.
        if bom and affected_components:
            for prod, comp_map in bom.items():
                # If product not in sales, still okay but won’t affect revenue calc unless it’s sold
                max_red = 0.0
                for comp in (comp_map or {}).keys():
                    if comp in affected_components:
                        max_red = max(max_red, _clamp01(affected_components[comp], default=0.0))
                if max_red > 0:
                    # merge (take worst-case if already set)
                    product_reduction[prod] = max(product_reduction.get(prod, 0.0), max_red)

        if not product_reduction:
            notes.append("No affected products/components provided; assuming disruption affects all products proportionally to severity.")
            for prod in sales.keys():
                product_reduction[prod] = sev  # fallback: everything reduced by severity

        # ---- Monte Carlo simulation ----
        random.seed(seed)

        # Model duration uncertainty: triangular around expected duration
        # (realistic: most likely near expected, sometimes shorter/longer)
        def sample_duration_days() -> int:
            low = max(1, int(dur * 0.6))
            mode = dur
            high = max(low, int(dur * 1.6))
            # triangular distribution
            u = random.random()
            c = (mode - low) / (high - low) if high > low else 0.5
            if u < c:
                return low + int(math.sqrt(u * (high - low) * (mode - low)))
            return high - int(math.sqrt((1 - u) * (high - low) * (high - mode)))

        # Translate severity and product reduction into effective shortfall
        # Example: severity=0.7 and product reduction=0.8 -> effective reduction ~ 0.56 (multiplicative)
        def effective_reduction(prod: str) -> float:
            red = _clamp01(product_reduction.get(prod, 0.0), default=0.0)
            return _clamp01(sev * red, default=0.0)

        total_rev_loss_if_disrupted = 0.0
        total_margin_loss_if_disrupted = 0.0

        for _ in range(int(n_trials)):
            d_days = sample_duration_days()
            d_days = min(d_days, H)  # disruption can’t exceed horizon in this model

            trial_rev_loss = 0.0
            trial_margin_loss = 0.0

            for prod, rev_per_day in sales.items():
                red = effective_reduction(prod)

                # inventory buffer reduces the disruption effect for some days
                buf = max(0.0, _safe_float(buffer_days.get(prod, 0.0), 0.0))
                covered_days = min(d_days, int(buf))

                # days exposed after inventory buffer
                exposed_days = max(0, d_days - covered_days)

                # demand shortfall fraction from supply reduction
                # revenue at risk = exposed_days * rev/day * reduction
                raw_loss = exposed_days * rev_per_day * red

                # allow mitigation via expedite/alternate sourcing
                recovered = raw_loss * expedite_fill
                remaining_loss = raw_loss - recovered

                # optional: expedite cost penalty (cost of recovering revenue)
                # You can store this separately; here we reduce margin via expedite cost.
                expedite_cost = recovered * expedite_cost_rate

                trial_rev_loss += remaining_loss
                margin_frac = _clamp01(margins.get(prod, default_margin), default=default_margin)
                trial_margin_loss += remaining_loss * margin_frac + expedite_cost

            total_rev_loss_if_disrupted += trial_rev_loss
            total_margin_loss_if_disrupted += trial_margin_loss

        if int(n_trials) <= 0:
            notes.append("n_trials <= 0; no simulation performed.")
            rev_if = 0.0
            mar_if = 0.0
        else:
            rev_if = total_rev_loss_if_disrupted / n_trials
            mar_if = total_margin_loss_if_disrupted / n_trials

        expected_rev = rev_if * p
        expected_mar = mar_if * p

        # fractions vs annual revenue (for interpretable %)
        frac_if = (rev_if / annual_rev) if annual_rev > 0 else 0.0
        frac_exp = (expected_rev / annual_rev) if annual_rev > 0 else 0.0

        # top affected products (by expected daily loss proxy)
        ranked: List[Tuple[str, float]] = []
        for prod, rev_per_day in sales.items():
            ranked.append((prod, rev_per_day * effective_reduction(prod)))
        ranked.sort(key=lambda x: x[1], reverse=True)
        top5 = ranked[:5]

        return RevenueAtRiskResult(
            timestamp_utc=_utc_iso(),
            horizon_days=H,
            n_trials=int(n_trials),
            expected_revenue_at_risk=float(expected_rev),
            expected_margin_at_risk=float(expected_mar),
            revenue_at_risk_if_disrupted=float(rev_if),
            margin_at_risk_if_disrupted=float(mar_if),
            disruption_probability=p,
            fraction_of_revenue_at_risk_if_disrupted=float(max(0.0, frac_if)),
            fraction_of_revenue_at_risk_expected=float(max(0.0, frac_exp)),
            top_affected_products=top5,
            notes=notes,
        )