# src/transparency/override_policy.py
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class OverridePolicy:
    """
    Human override thresholds:
      - if decision impact is high
      - if risk is critical
      - if constraints violated
      - if bias checks show issues
    """
    auto_approve_threshold: float = 0.80
    human_review_threshold: float = 0.50
    executive_review_threshold: float = 0.30

    # hard guardrails
    max_single_supplier_allocation_pct: float = 0.70
    max_budget_fraction_of_revenue: float = 0.05  # 5% of annual revenue
    min_service_level_floor: float = 0.85

    def decide_approval_mode(
        self,
        recommendation_score: float,
        has_high_severity_violation: bool,
        estimated_cost: float,
        annual_revenue: float,
        service_p05: float
    ) -> Dict[str, Any]:
        # Guardrails override everything
        if has_high_severity_violation:
            return {"mode": "executive_approval", "reason": "High severity constraint violation detected."}

        if annual_revenue > 0 and estimated_cost > self.max_budget_fraction_of_revenue * annual_revenue:
            return {"mode": "executive_approval", "reason": "Cost exceeds budget guardrail."}

        if service_p05 < self.min_service_level_floor:
            return {"mode": "human_approval", "reason": "Service tail-risk below floor; requires review."}

        # Score-based mode
        if recommendation_score >= self.auto_approve_threshold:
            return {"mode": "automatic", "reason": "High confidence recommendation."}
        if recommendation_score >= self.human_review_threshold:
            return {"mode": "human_approval", "reason": "Moderate confidence recommendation."}
        return {"mode": "executive_approval", "reason": "Low confidence recommendation."}