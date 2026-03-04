from __future__ import annotations

from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import numpy as np
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ImpactLevel(Enum):
    SEVERE = "severe"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"


@dataclass
class RiskScore:
    overall_score: float
    risk_level: RiskLevel
    components: Dict[str, float]
    confidence: float
    last_updated: datetime
    trend: str = "stable"


@dataclass
class ImpactScore:
    overall_score: float
    impact_level: ImpactLevel
    revenue_impact: float
    operational_impact: float
    customer_impact: float
    cost_impact: float
    confidence: float
    last_updated: datetime


class ScoringSystem:
    def __init__(self) -> None:
        # Scoring thresholds
        self.risk_thresholds = {
            RiskLevel.CRITICAL: 0.8,
            RiskLevel.HIGH: 0.6,
            RiskLevel.MEDIUM: 0.4,
            RiskLevel.LOW: 0.0,
        }

        self.impact_thresholds = {
            ImpactLevel.SEVERE: 0.8,
            ImpactLevel.HIGH: 0.6,
            ImpactLevel.MODERATE: 0.4,
            ImpactLevel.LOW: 0.0,
        }

        # Component weights for risk scoring (should sum to 1.0)
        self.risk_component_weights = {
            "disruption_risk": 0.25,
            "supplier_risk": 0.20,
            "operational_risk": 0.20,
            "geographic_risk": 0.15,
            "financial_risk": 0.10,
            "market_risk": 0.10,
        }

        # Impact component weights (should sum to 1.0)
        self.impact_component_weights = {
            "revenue_impact": 0.35,
            "operational_impact": 0.25,
            "customer_impact": 0.20,
            "cost_impact": 0.20,
        }

    # -------------------------
    # Core scoring (0..1)
    # -------------------------
    def calculate_risk_score(self, components: Dict[str, float], confidence: float = 0.8) -> RiskScore:
        """Calculate overall risk score from components (all clamped to [0,1])."""
        validated_components: Dict[str, float] = {}
        for component in self.risk_component_weights:
            value = float(components.get(component, 0.0))
            if not np.isfinite(value):
                value = 0.0
            validated_components[component] = max(0.0, min(1.0, value))

        overall_score = sum(
            validated_components[component] * weight
            for component, weight in self.risk_component_weights.items()
        )
        overall_score = float(max(0.0, min(1.0, overall_score)))

        risk_level = self._determine_risk_level(overall_score)

        return RiskScore(
            overall_score=overall_score,
            risk_level=risk_level,
            components=validated_components,
            confidence=float(max(0.0, min(1.0, confidence))),
            last_updated=datetime.now(timezone.utc),
            trend="stable",
        )

    def calculate_impact_score(self, components: Dict[str, float], confidence: float = 0.8) -> ImpactScore:
        """Calculate overall impact score from components (all clamped to [0,1])."""
        validated_components: Dict[str, float] = {}
        for component in self.impact_component_weights:
            value = float(components.get(component, 0.0))
            if not np.isfinite(value):
                value = 0.0
            validated_components[component] = max(0.0, min(1.0, value))

        overall_score = sum(
            validated_components[component] * weight
            for component, weight in self.impact_component_weights.items()
        )
        overall_score = float(max(0.0, min(1.0, overall_score)))

        impact_level = self._determine_impact_level(overall_score)

        return ImpactScore(
            overall_score=overall_score,
            impact_level=impact_level,
            revenue_impact=validated_components.get("revenue_impact", 0.0),
            operational_impact=validated_components.get("operational_impact", 0.0),
            customer_impact=validated_components.get("customer_impact", 0.0),
            cost_impact=validated_components.get("cost_impact", 0.0),
            confidence=float(max(0.0, min(1.0, confidence))),
            last_updated=datetime.now(timezone.utc),
        )

    # -------------------------
    # Sub-scorers
    # -------------------------
    def calculate_supplier_risk_score(self, supplier_data: Dict[str, Any]) -> float:
        """Calculate risk score for an individual supplier (0..1)."""
        fin_health = float(supplier_data.get("financial_health_score", 0.7))
        perf = float(supplier_data.get("performance_score", 0.8))
        geo = float(supplier_data.get("geopolitical_risk_score", 0.3))
        divers = float(supplier_data.get("diversification_level", 0.5))

        for vname, v in [("financial_health_score", fin_health), ("performance_score", perf),
                         ("geopolitical_risk_score", geo), ("diversification_level", divers)]:
            if not np.isfinite(v):
                logger.warning("Non-finite %s in supplier_data; treating as 0.0", vname)

        fin_health = max(0.0, min(1.0, fin_health))
        perf = max(0.0, min(1.0, perf))
        geo = max(0.0, min(1.0, geo))
        divers = max(0.0, min(1.0, divers))

        financial_risk = 1.0 - fin_health
        performance_risk = 1.0 - perf
        geographic_risk = geo
        concentration_risk = 1.0 - divers

        supplier_risk = (
            financial_risk * 0.3
            + performance_risk * 0.25
            + geographic_risk * 0.25
            + concentration_risk * 0.2
        )
        return float(max(0.0, min(1.0, supplier_risk)))

    def calculate_disruption_risk_score(self, disruption_data: Dict[str, Any]) -> float:
        """Calculate disruption risk score (severity * probability * relevance), clamped to 0..1."""
        severity = float(disruption_data.get("severity_score", 0.5))
        probability = float(disruption_data.get("probability_of_occurrence", 0.5))
        relevance = float(disruption_data.get("relevance_score", 0.5))

        # clamp + protect against NaN/inf
        def _clamp01(x: float) -> float:
            if not np.isfinite(x):
                return 0.0
            return max(0.0, min(1.0, x))

        severity = _clamp01(severity)
        probability = _clamp01(probability)
        relevance = _clamp01(relevance)

        disruption_risk = severity * probability * relevance
        return float(max(0.0, min(1.0, disruption_risk)))

    def calculate_portfolio_risk_score(self, supplier_scores: List[float], weights: Optional[List[float]] = None) -> float:
        """Calculate portfolio risk score from multiple suppliers, with concentration penalty."""
        if not supplier_scores:
            return 0.0

        # Clamp supplier scores
        cleaned_scores: List[float] = []
        for s in supplier_scores:
            s = float(s)
            if not np.isfinite(s):
                s = 0.0
            cleaned_scores.append(max(0.0, min(1.0, s)))

        n = len(cleaned_scores)

        if weights is None or len(weights) != n:
            weights = [1.0 / n] * n
        else:
            weights = [float(w) if np.isfinite(float(w)) else 0.0 for w in weights]

        # No negative weights (avoid weirdness)
        weights = [max(0.0, w) for w in weights]

        total_weight = sum(weights)
        if total_weight <= 0:
            weights = [1.0 / n] * n
        else:
            weights = [w / total_weight for w in weights]

        portfolio_risk = sum(score * weight for score, weight in zip(cleaned_scores, weights))
        concentration_penalty = self._calculate_concentration_penalty(weights)
        final_risk = portfolio_risk + concentration_penalty

        return float(max(0.0, min(1.0, final_risk)))

    # -------------------------
    # Helpers
    # -------------------------
    def _calculate_concentration_penalty(self, weights: List[float]) -> float:
        """HHI-based concentration penalty. 0.25 ~ 4 equal suppliers baseline."""
        hhi = sum(w * w for w in weights)
        penalty = (hhi - 0.25) * 0.2
        return float(max(0.0, penalty))

    def _determine_risk_level(self, score: float) -> RiskLevel:
        if score >= self.risk_thresholds[RiskLevel.CRITICAL]:
            return RiskLevel.CRITICAL
        if score >= self.risk_thresholds[RiskLevel.HIGH]:
            return RiskLevel.HIGH
        if score >= self.risk_thresholds[RiskLevel.MEDIUM]:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def _determine_impact_level(self, score: float) -> ImpactLevel:
        if score >= self.impact_thresholds[ImpactLevel.SEVERE]:
            return ImpactLevel.SEVERE
        if score >= self.impact_thresholds[ImpactLevel.HIGH]:
            return ImpactLevel.HIGH
        if score >= self.impact_thresholds[ImpactLevel.MODERATE]:
            return ImpactLevel.MODERATE
        return ImpactLevel.LOW

    def calculate_trend(self, historical_scores: List[float], window_size: int = 5) -> str:
        """Robust trend: handles NaN/inf and catches polyfit issues."""
        if len(historical_scores) < 2:
            return "stable"

        recent = historical_scores[-window_size:] if len(historical_scores) >= window_size else historical_scores
        y = np.array([float(v) for v in recent], dtype=float)

        # Remove non-finite values
        finite_mask = np.isfinite(y)
        y = y[finite_mask]
        if y.size < 2:
            return "stable"

        x = np.arange(y.size, dtype=float)

        try:
            slope = float(np.polyfit(x, y, 1)[0])
        except Exception as e:
            logger.warning("Trend polyfit failed: %s", e)
            return "stable"

        if slope > 0.02:
            return "increasing"
        if slope < -0.02:
            return "decreasing"
        return "stable"

    def calculate_volatility(self, historical_scores: List[float], window_size: int = 10) -> float:
        """Volatility = std dev of recent finite scores."""
        if len(historical_scores) < 2:
            return 0.0

        recent = historical_scores[-window_size:] if len(historical_scores) >= window_size else historical_scores
        y = np.array([float(v) for v in recent], dtype=float)
        y = y[np.isfinite(y)]
        if y.size < 2:
            return 0.0

        return float(np.std(y))

    def normalize_score(self, raw_score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Normalize raw score to 0-1 range safely."""
        raw_score = float(raw_score)
        if not np.isfinite(raw_score) or max_val <= min_val:
            return 0.0
        normalized = (raw_score - min_val) / (max_val - min_val)
        return float(max(0.0, min(1.0, normalized)))

    def aggregate_risk_scores(self, scores: List[RiskScore], method: str = "weighted_average") -> RiskScore:
        """Aggregate multiple RiskScores."""
        if not scores:
            raise ValueError("No scores to aggregate")

        if method == "maximum":
            return max(scores, key=lambda s: s.overall_score)

        if method != "weighted_average":
            raise ValueError(f"Unknown aggregation method: {method}")

        # Weight by confidence
        confs = [float(s.confidence) if np.isfinite(float(s.confidence)) else 0.0 for s in scores]
        total = sum(confs)
        if total <= 0:
            weights = [1.0 / len(scores)] * len(scores)
        else:
            weights = [c / total for c in confs]

        aggregated_components: Dict[str, float] = {}
        for component in self.risk_component_weights:
            vals = [float(s.components.get(component, 0.0)) for s in scores]
            vals = [0.0 if not np.isfinite(v) else max(0.0, min(1.0, v)) for v in vals]
            aggregated_components[component] = float(sum(v * w for v, w in zip(vals, weights)))

        overall_score = float(sum(float(s.overall_score) * w for s, w in zip(scores, weights)))
        overall_score = float(max(0.0, min(1.0, overall_score)))

        aggregated_confidence = float(sum(float(s.confidence) * w for s, w in zip(scores, weights)))
        aggregated_confidence = float(max(0.0, min(1.0, aggregated_confidence)))

        return RiskScore(
            overall_score=overall_score,
            risk_level=self._determine_risk_level(overall_score),
            components=aggregated_components,
            confidence=aggregated_confidence,
            last_updated=datetime.now(timezone.utc),
            trend="aggregated",
        )