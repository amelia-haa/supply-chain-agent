from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime

@dataclass
class NewsItem:
    id: str
    ts: str
    title: str
    url: str
    source_country: str
    source: str  # gdelt

@dataclass
class DisasterItem:
    id: str
    ts: str
    event_type: str
    alert_level: str
    country: str
    name: str
    source: str  # gdacs

@dataclass
class RiskSignal:
    id: str
    ts: str
    signal_type: str   # news | disaster | erp | supplier
    category: str      # shipping | geopolitics | supplier | climate | demand | capacity | quality | finance | cyber | other
    region: str
    severity: float    # 0..1
    confidence: float  # 0..1
    summary: str
    evidence: Dict[str, Any]
    tags: List[str]

@dataclass
class ErpSignal:
    ts: str
    sku: str
    supplier_id: str
    plant: str
    on_hand: float
    daily_demand: float
    lead_time_days: int
    open_po_qty: float
    next_delivery_eta: Optional[str]
    backlog_qty: float
    late_shipments_30d: int
    quality_incidents_90d: int

@dataclass
class SupplierHealth:
    supplier_id: str
    name: str
    region: str
    score: float        # 0..1 (higher = healthier)
    risk_score: float   # 0..1 (higher = riskier)
    drivers: List[str]
    raw: Dict[str, Any]
    updated_at: str