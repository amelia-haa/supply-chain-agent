from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

@dataclass
class Company:
    id: str
    name: str
    industry: str
    location: str

    supplier_concentration_risk: float = 0.6
    regional_exposure_score: float = 0.6
    lead_time_sensitivity: float = 0.6
    inventory_buffer_policy: float = 0.3
    risk_appetite: float = 0.5
    service_level_target: float = 0.95

    annual_revenue: float = 50_000_000  # optional but helpful

@dataclass
class Supplier:
    id: str
    name: str
    region: str
    reliability_score: float = 0.8
    on_time_delivery_rate: float = 0.9
    quality_score: float = 0.9
    financial_health_score: float = 0.8
    geopolitical_risk_score: float = 0.3
    climate_risk_score: float = 0.2
    lead_time_days: int = 30
    unit_cost: float = 10.0
    capacity_units_per_month: int = 100_000
    criticality_score: float = 0.5
    risk_score: float = 0.3  # higher = worse

@dataclass
class Disruption:
    title: str
    description: str
    disruption_type: str
    severity_score: float
    id: str = field(default_factory=lambda: f"DSP_{int(datetime.utcnow().timestamp())}")
    affected_regions: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class RiskAssessment:
    company_id: str
    composite_risk_score: float
    key_drivers: List[str] = field(default_factory=list)
    critical_suppliers: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def overall_risk_score(self) -> float:
        return self.composite_risk_score

@dataclass
class MitigationAction:
    company_id: str
    risk_assessment_id: str
    action_type: str  # resourcing/buffering/rerouting/negotiation/escalation
    title: str
    description: str
    priority_level: str  # low/medium/high/critical

    estimated_cost: float
    estimated_benefit: float
    roi_score: float
    implementation_time_days: int
    urgency_score: float = 0.5

    id: str = field(default_factory=lambda: f"ACT_{int(datetime.utcnow().timestamp())}")
