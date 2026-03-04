# src/transparency/risk_justifier.py
from typing import Dict, Any, List
import numpy as np
from agent.models import Company, Supplier

class RiskJustifier:
    """
    Produces risk justification logic:
      - what factors drive risk
      - what data supports it
      - how recommended actions reduce it
    """

    async def justify_company_risk(
        self,
        company: Company,
        suppliers: List[Supplier],
        scenario_summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        avg_supplier_risk = float(np.mean([s.risk_score for s in suppliers])) if suppliers else 0.3
        avg_reliability = float(np.mean([s.reliability_score for s in suppliers])) if suppliers else 0.8

        service_p05 = scenario_summary.get("service", {}).get("p05", company.service_level_target)
        cost_p95 = scenario_summary.get("cost_multiplier", {}).get("p95", 1.0)

        # Composite “explainable” drivers (not a perfect model, but transparent)
        drivers = []

        if company.supplier_concentration_risk > 0.6:
            drivers.append({
                "driver": "supplier_concentration",
                "evidence": {"supplier_concentration_risk": company.supplier_concentration_risk},
                "why_it_matters": "High dependence reduces fallback options if a supplier fails."
            })

        if company.regional_exposure_score > 0.6:
            drivers.append({
                "driver": "regional_exposure",
                "evidence": {"regional_exposure_score": company.regional_exposure_score},
                "why_it_matters": "Regional shocks can affect multiple suppliers/logistics routes simultaneously."
            })

        drivers.append({
            "driver": "supplier_risk_and_reliability",
            "evidence": {"avg_supplier_risk": avg_supplier_risk, "avg_reliability": avg_reliability},
            "why_it_matters": "Higher supplier risk or low reliability increases disruption probability."
        })

        drivers.append({
            "driver": "scenario_tail_risk",
            "evidence": {"service_p05": service_p05, "cost_p95": cost_p95},
            "why_it_matters": "Worst-case scenarios determine whether service collapses under stress."
        })

        # Risk justification summary
        justification = {
            "risk_statement": "Supply chain risk is elevated due to concentration/exposure + supplier risk + scenario tail risk.",
            "drivers": drivers,
            "risk_signals": {
                "service_p05": service_p05,
                "cost_p95": cost_p95,
                "supplier_concentration_risk": company.supplier_concentration_risk,
                "regional_exposure_score": company.regional_exposure_score,
                "avg_supplier_risk": avg_supplier_risk
            },
            "recommended_logic": [
                "If service tail-risk is low -> prioritize expediting, buffering, reallocation.",
                "If cost tail-risk is high but service ok -> reduce expediting and normalize shipments.",
                "If concentration is high -> diversify suppliers and split POs."
            ]
        }

        return justification