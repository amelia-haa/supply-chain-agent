# src/action/purchase_order_adjuster.py
"""
Purchase Order Adjustment Suggestions (Planning & Autonomous Action Layer)

What this module does:
- Pulls OPEN purchase orders (POs) from ERPIntegrator (real ERP in prod; mock OK for dev)
- Scores each PO for disruption/late/stockout risk
- Generates adjustment suggestions:
    * EXPEDITE / DEFER / CANCEL / INCREASE_QTY / DECREASE_QTY / SPLIT_ORDER / SWITCH_SUPPLIER
- Estimates cost delta + service-level impact (simple but extensible)
- Optionally uses Gemini (LLM) to generate human-readable rationales
- Returns structured suggestions suitable for:
    * human approval workflow
    * automatic execution (if confidence high enough)

This is “comprehensive” in the sense it includes:
- risk scoring
- decision policy
- cost/service trade-off estimation
- supplier reassignment option
- approvals + execution hooks (via ERPIntegrator / WorkflowManager / EmailGenerator)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import google.generativeai as genai

from agent.config import settings
from agent.models import Company, Supplier  # keep consistent with your project
import logging

logger = logging.getLogger(__name__)


# ----------------------------
# Data contracts (PO + Suggestions)
# ----------------------------

class POAction(str, Enum):
    NO_CHANGE = "no_change"
    EXPEDITE = "expedite"
    DEFER = "defer"
    CANCEL = "cancel"
    INCREASE_QTY = "increase_qty"
    DECREASE_QTY = "decrease_qty"
    SPLIT_ORDER = "split_order"
    SWITCH_SUPPLIER = "switch_supplier"
    EXPEDITE_AND_INCREASE = "expedite_and_increase"


@dataclass
class PurchaseOrder:
    po_id: str
    supplier_id: str
    supplier_name: str
    item_sku: str
    item_name: str
    qty: float
    unit_cost: float
    currency: str
    created_at: datetime
    requested_delivery_date: datetime
    promised_delivery_date: datetime
    incoterm: str = "FOB"
    ship_mode: str = "ocean"  # ocean/air/ground/rail
    region_from: str = "unknown"
    region_to: str = "unknown"
    status: str = "open"  # open/confirmed/shipped/received/cancelled

    # optional planning inputs (if you have them in ERP)
    on_hand: Optional[float] = None
    daily_demand_forecast: Optional[float] = None
    safety_stock_target: Optional[float] = None
    min_order_qty: Optional[float] = None


@dataclass
class POSuggestion:
    po_id: str
    action: POAction
    confidence: float  # 0..1
    priority: str      # low/medium/high/critical

    # proposed edits
    recommended_qty: Optional[float] = None
    recommended_promised_delivery_date: Optional[datetime] = None
    recommended_ship_mode: Optional[str] = None
    recommended_supplier_id: Optional[str] = None
    recommended_supplier_name: Optional[str] = None
    split_plan: Optional[List[Dict[str, Any]]] = None  # e.g., [{qty, ship_mode, promised_date}, ...]

    # quant impacts (estimates)
    estimated_cost_delta: float = 0.0
    estimated_service_level_delta: float = 0.0
    estimated_late_risk_delta: float = 0.0
    estimated_stockout_risk_delta: float = 0.0

    reasons: List[str] = None
    llm_rationale: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # datetime serialization
        if d.get("recommended_promised_delivery_date"):
            d["recommended_promised_delivery_date"] = d["recommended_promised_delivery_date"].isoformat()
        return d


# ----------------------------
# ERP + workflow interfaces (duck-typed)
# ----------------------------

class ERPIntegratorProtocol:
    async def get_open_purchase_orders(self, company: Company) -> List[Dict[str, Any]]:
        raise NotImplementedError

    async def apply_po_changes(self, company: Company, changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        raise NotImplementedError


class WorkflowManagerProtocol:
    async def create_po_approval_workflow(self, company: Company, suggestions: List[Dict[str, Any]]) -> Dict[str, Any]:
        raise NotImplementedError


class EmailGeneratorProtocol:
    async def generate_po_adjustment_email(self, company: Company, suggestions: List[Dict[str, Any]]) -> Dict[str, Any]:
        raise NotImplementedError


# ----------------------------
# Purchase Order Adjustment Engine
# ----------------------------

class PurchaseOrderAdjustmentEngine:
    def __init__(
        self,
        erp_integrator: ERPIntegratorProtocol,
        *,
        workflow_manager: Optional[WorkflowManagerProtocol] = None,
        email_generator: Optional[EmailGeneratorProtocol] = None,
        enable_llm_rationales: bool = True,
    ):
        self.erp_integrator = erp_integrator
        self.workflow_manager = workflow_manager
        self.email_generator = email_generator

        self.enable_llm_rationales = enable_llm_rationales

        genai.configure(api_key=settings.google_api_key)
        self.model = genai.GenerativeModel(settings.gemini_model)

        # Policy knobs (tune to your agent)
        self.thresholds = {
            "auto_execute_confidence": 0.85,
            "high_priority_risk": 0.75,
            "critical_priority_risk": 0.88,
        }

        # Shipping multipliers for expedite / mode switch
        self.ship_mode_cost_multiplier = {
            "ocean": 1.00,
            "ground": 1.05,
            "rail": 1.08,
            "air": 1.35,  # expedite
        }
        self.ship_mode_lead_time_multiplier = {
            "ocean": 1.00,
            "ground": 0.70,
            "rail": 0.75,
            "air": 0.35,
        }

    # -------- public API --------

    async def generate_po_adjustment_suggestions(
        self,
        company: Company,
        suppliers: List[Supplier],
        *,
        horizon_days: int = 90,
        max_suggestions: int = 25,
        allow_supplier_switch: bool = True,
        allow_split_orders: bool = True,
        include_llm_rationales: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Main entry point:
        - Fetch open POs
        - Compute risk + stockout risk
        - Propose actions
        - Optionally generate LLM rationales
        """
        include_llm = self.enable_llm_rationales if include_llm_rationales is None else include_llm_rationales

        raw_pos = await self.erp_integrator.get_open_purchase_orders(company)
        pos = [self._coerce_po(po) for po in raw_pos if self._within_horizon(po, horizon_days)]

        if not pos:
            return {
                "company_id": getattr(company, "id", None),
                "generated_at": datetime.utcnow().isoformat(),
                "total_open_pos_considered": 0,
                "suggestions": [],
                "summary": {"message": "No open POs found within horizon."},
            }

        supplier_index = self._build_supplier_index(suppliers)

        suggestions: List[POSuggestion] = []
        for po in pos:
            features = self._compute_po_features(company, po, supplier_index)
            suggestion = self._propose_adjustment(
                company,
                po,
                features,
                suppliers,
                allow_supplier_switch=allow_supplier_switch,
                allow_split_orders=allow_split_orders,
            )
            if suggestion.action != POAction.NO_CHANGE:
                suggestions.append(suggestion)

        # rank suggestions (highest priority + confidence first)
        suggestions.sort(
            key=lambda s: (
                {"critical": 3, "high": 2, "medium": 1, "low": 0}.get(s.priority, 0),
                s.confidence,
            ),
            reverse=True,
        )
        suggestions = suggestions[:max_suggestions]

        # LLM rationales (parallelized)
        if include_llm and suggestions:
            await self._attach_llm_rationales(company, suggestions)

        result = {
            "company_id": getattr(company, "id", None),
            "generated_at": datetime.utcnow().isoformat(),
            "total_open_pos_considered": len(pos),
            "suggestions": [s.to_dict() for s in suggestions],
            "summary": self._summarize(suggestions),
        }
        return result

    async def route_suggestions_to_workflow(
        self,
        company: Company,
        suggestion_payload: Dict[str, Any],
        *,
        always_require_approval: bool = True,
    ) -> Dict[str, Any]:
        """
        Optional helper:
        - Create approval workflow (if workflow manager exists)
        - Generate an internal PO adjustment email (if email generator exists)
        """
        suggestions = suggestion_payload.get("suggestions", [])

        workflow_result = None
        if self.workflow_manager:
            workflow_result = await self.workflow_manager.create_po_approval_workflow(company, suggestions)

        email_result = None
        if self.email_generator:
            email_result = await self.email_generator.generate_po_adjustment_email(company, suggestions)

        # Decide auto-execute subset if allowed
        auto_execute = []
        if not always_require_approval:
            for s in suggestions:
                if float(s.get("confidence", 0)) >= self.thresholds["auto_execute_confidence"]:
                    auto_execute.append(s)

        return {
            "workflow_result": workflow_result,
            "email_result": email_result,
            "auto_execute_candidates": auto_execute,
        }

    async def apply_suggestions(
        self,
        company: Company,
        suggestions: List[Dict[str, Any]],
        *,
        auto_execute_only: bool = True,
    ) -> Dict[str, Any]:
        """
        Applies suggestions via ERP integrator.
        If auto_execute_only=True, only applies suggestions above confidence threshold.
        """
        changes = []
        for s in suggestions:
            if auto_execute_only and float(s.get("confidence", 0)) < self.thresholds["auto_execute_confidence"]:
                continue
            changes.append(self._suggestion_to_erp_change(s))

        if not changes:
            return {"applied": False, "message": "No suggestions eligible for execution.", "changes": []}

        return await self.erp_integrator.apply_po_changes(company, changes)

    # -------- coercion + filters --------

    def _coerce_po(self, raw: Dict[str, Any]) -> PurchaseOrder:
        # expect ERP to provide these keys; fallback hard if missing
        return PurchaseOrder(
            po_id=str(raw["po_id"]),
            supplier_id=str(raw["supplier_id"]),
            supplier_name=str(raw.get("supplier_name", raw["supplier_id"])),
            item_sku=str(raw["item_sku"]),
            item_name=str(raw.get("item_name", raw["item_sku"])),
            qty=float(raw["qty"]),
            unit_cost=float(raw["unit_cost"]),
            currency=str(raw.get("currency", "USD")),
            created_at=self._to_dt(raw.get("created_at", datetime.utcnow())),
            requested_delivery_date=self._to_dt(raw.get("requested_delivery_date", datetime.utcnow() + timedelta(days=30))),
            promised_delivery_date=self._to_dt(raw.get("promised_delivery_date", datetime.utcnow() + timedelta(days=30))),
            incoterm=str(raw.get("incoterm", "FOB")),
            ship_mode=str(raw.get("ship_mode", "ocean")),
            region_from=str(raw.get("region_from", "unknown")),
            region_to=str(raw.get("region_to", "unknown")),
            status=str(raw.get("status", "open")),
            on_hand=raw.get("on_hand", None),
            daily_demand_forecast=raw.get("daily_demand_forecast", None),
            safety_stock_target=raw.get("safety_stock_target", None),
            min_order_qty=raw.get("min_order_qty", None),
        )

    def _within_horizon(self, raw: Dict[str, Any], horizon_days: int) -> bool:
        promised = self._to_dt(raw.get("promised_delivery_date", datetime.utcnow() + timedelta(days=9999)))
        return promised <= datetime.utcnow() + timedelta(days=horizon_days)

    def _to_dt(self, x: Any) -> datetime:
        if isinstance(x, datetime):
            return x
        if isinstance(x, str):
            try:
                return datetime.fromisoformat(x.replace("Z", "+00:00")).replace(tzinfo=None)
            except Exception:
                pass
        # fallback
        return datetime.utcnow()

    # -------- feature engineering --------

    def _build_supplier_index(self, suppliers: List[Supplier]) -> Dict[str, Supplier]:
        idx = {}
        for s in suppliers:
            sid = str(getattr(s, "id", getattr(s, "supplier_id", "")))
            if sid:
                idx[sid] = s
        return idx

    def _compute_po_features(
        self,
        company: Company,
        po: PurchaseOrder,
        supplier_index: Dict[str, Supplier],
    ) -> Dict[str, float]:
        """
        Returns normalized features 0..1 where possible.
        """
        now = datetime.utcnow()
        days_to_promise = max(0.0, (po.promised_delivery_date - now).total_seconds() / 86400.0)
        days_past_request = max(0.0, (po.promised_delivery_date - po.requested_delivery_date).total_seconds() / 86400.0)

        # Supplier risk input (use your Supplier model fields if available)
        supplier_obj = supplier_index.get(po.supplier_id)
        supplier_risk = float(getattr(supplier_obj, "risk_score", 0.5)) if supplier_obj else 0.55
        supplier_criticality = float(getattr(supplier_obj, "criticality_score", 0.5)) if supplier_obj else 0.5
        supplier_reliability = float(getattr(supplier_obj, "reliability_score", 0.7)) if supplier_obj else 0.7

        # Shipping risk heuristic (replace with real lane risk data)
        ship_mode_risk = {"ocean": 0.55, "ground": 0.35, "rail": 0.40, "air": 0.25}.get(po.ship_mode, 0.5)

        # Inventory/stockout risk (if ERP provides on_hand + forecast + safety stock)
        stockout_risk = 0.5
        if po.on_hand is not None and po.daily_demand_forecast is not None:
            runway_days = po.on_hand / max(1e-9, float(po.daily_demand_forecast))
            target = float(po.safety_stock_target) if po.safety_stock_target is not None else 14.0
            # if runway below target -> higher risk
            stockout_risk = float(np.clip((target - runway_days) / max(1.0, target), 0.0, 1.0))

        # Late risk: if promised is already behind requested, or promise is soon with risky supplier
        lateness_pressure = float(np.clip(days_past_request / 30.0, 0.0, 1.0))
        time_pressure = float(np.clip((30.0 - days_to_promise) / 30.0, 0.0, 1.0))

        # composite risk score (tunable)
        composite_risk = (
            0.35 * supplier_risk +
            0.20 * (1.0 - supplier_reliability) +
            0.20 * ship_mode_risk +
            0.15 * stockout_risk +
            0.10 * time_pressure
        )
        composite_risk = float(np.clip(composite_risk, 0.0, 1.0))

        return {
            "days_to_promise": float(days_to_promise),
            "days_past_request": float(days_past_request),
            "supplier_risk": float(np.clip(supplier_risk, 0.0, 1.0)),
            "supplier_criticality": float(np.clip(supplier_criticality, 0.0, 1.0)),
            "supplier_reliability": float(np.clip(supplier_reliability, 0.0, 1.0)),
            "ship_mode_risk": float(np.clip(ship_mode_risk, 0.0, 1.0)),
            "stockout_risk": float(np.clip(stockout_risk, 0.0, 1.0)),
            "lateness_pressure": float(np.clip(lateness_pressure, 0.0, 1.0)),
            "time_pressure": float(np.clip(time_pressure, 0.0, 1.0)),
            "composite_risk": composite_risk,
        }

    # -------- recommendation policy --------

    def _priority_from_risk(self, r: float) -> str:
        if r >= self.thresholds["critical_priority_risk"]:
            return "critical"
        if r >= self.thresholds["high_priority_risk"]:
            return "high"
        if r >= 0.55:
            return "medium"
        return "low"

    def _propose_adjustment(
        self,
        company: Company,
        po: PurchaseOrder,
        f: Dict[str, float],
        suppliers: List[Supplier],
        *,
        allow_supplier_switch: bool,
        allow_split_orders: bool,
    ) -> POSuggestion:
        """
        Produces a single best suggestion per PO (simple policy).
        You can extend to output multiple candidate actions per PO.
        """
        reasons: List[str] = []
        r = f["composite_risk"]
        priority = self._priority_from_risk(r)

        # quick signals
        high_stockout = f["stockout_risk"] >= 0.65
        risky_supplier = f["supplier_risk"] >= 0.70
        low_reliability = f["supplier_reliability"] <= 0.55
        high_ship_risk = f["ship_mode_risk"] >= 0.60
        time_pressure = f["time_pressure"] >= 0.60

        # baseline: no change
        action = POAction.NO_CHANGE
        confidence = 0.45

        # Option A: expedite (mode switch to air) when time/stockout pressure high
        if (high_stockout or time_pressure) and po.ship_mode != "air":
            action = POAction.EXPEDITE
            confidence = 0.70 + 0.15 * float(high_stockout) + 0.10 * float(time_pressure)
            reasons.append("Time/stockout pressure suggests expediting to protect service level.")
            if high_stockout:
                reasons.append("Stockout risk is high based on on-hand vs demand forecast.")
            if time_pressure:
                reasons.append("Delivery window is tight relative to current promise date.")

        # Option B: expedite + increase when stockout is critical
        if high_stockout and f["stockout_risk"] >= 0.85:
            action = POAction.EXPEDITE_AND_INCREASE
            confidence = max(confidence, 0.82)
            reasons.append("Critical stockout risk: expedite and increase quantity to restore buffer.")

        # Option C: switch supplier when supplier is risky/unreliable AND alternatives exist
        if allow_supplier_switch and (risky_supplier or low_reliability) and priority in ("high", "critical"):
            alt = self._find_alternative_supplier(po, suppliers)
            if alt:
                action = POAction.SWITCH_SUPPLIER
                confidence = max(confidence, 0.78 + 0.10 * float(low_reliability))
                reasons.append("Supplier risk/reliability indicates re-sourcing to a safer supplier.")
                reasons.append(f"Found alternative supplier candidate: {alt[1]}.")

        # Option D: split order when risk high AND time pressure medium-high (hedge)
        if allow_split_orders and r >= 0.70 and (high_ship_risk or risky_supplier) and po.qty >= 2:
            action = POAction.SPLIT_ORDER
            confidence = max(confidence, 0.76)
            reasons.append("High risk: split order to hedge (partial expedite + partial standard).")

        # Option E: defer/cancel if demand low and inventory high (requires inputs)
        if po.on_hand is not None and po.daily_demand_forecast is not None:
            runway_days = po.on_hand / max(1e-9, float(po.daily_demand_forecast))
            if runway_days >= 60 and r < 0.55:
                # if runway very high, consider deferring
                action = POAction.DEFER
                confidence = max(confidence, 0.72)
                reasons.append("Inventory runway is high; deferring can reduce carrying costs with low risk.")

        # Build suggested edits + deltas
        suggestion = self._materialize_suggestion(company, po, f, action, confidence, priority, reasons, suppliers)
        return suggestion

    def _find_alternative_supplier(self, po: PurchaseOrder, suppliers: List[Supplier]) -> Optional[Tuple[str, str]]:
        """
        Very simple: pick best reliability among suppliers that can supply this SKU (if you track it).
        If you don't track SKUs per supplier yet, it picks the best reliability overall.
        """
        candidates = []
        for s in suppliers:
            sid = str(getattr(s, "id", getattr(s, "supplier_id", "")))
            name = str(getattr(s, "name", "Unknown Supplier"))
            reliability = float(getattr(s, "reliability_score", 0.7))
            risk = float(getattr(s, "risk_score", 0.5))
            # if you have catalog support, uncomment:
            # skus = set(getattr(s, "supported_skus", []))
            # if po.item_sku not in skus:
            #     continue
            candidates.append((sid, name, reliability, risk))

        if not candidates:
            return None

        # prefer high reliability, low risk
        candidates.sort(key=lambda x: (x[2], -x[3]), reverse=True)
        best = candidates[0]
        if best[0] == po.supplier_id:
            return None
        return (best[0], best[1])

    def _materialize_suggestion(
        self,
        company: Company,
        po: PurchaseOrder,
        f: Dict[str, float],
        action: POAction,
        confidence: float,
        priority: str,
        reasons: List[str],
        suppliers: List[Supplier],
    ) -> POSuggestion:
        confidence = float(np.clip(confidence, 0.0, 1.0))

        s = POSuggestion(
            po_id=po.po_id,
            action=action,
            confidence=confidence,
            priority=priority,
            reasons=reasons or [],
        )

        # Default deltas
        base_late_risk = f["composite_risk"]
        base_stockout = f["stockout_risk"]

        if action == POAction.NO_CHANGE:
            return s

        if action in (POAction.EXPEDITE, POAction.EXPEDITE_AND_INCREASE):
            s.recommended_ship_mode = "air"
            # estimate earlier promise date by scaling lead time
            now = datetime.utcnow()
            remaining = max(1.0, (po.promised_delivery_date - now).total_seconds() / 86400.0)
            new_remaining = remaining * self.ship_mode_lead_time_multiplier.get("air", 0.35)
            s.recommended_promised_delivery_date = now + timedelta(days=float(new_remaining))

            # cost delta from ship mode multiplier (simplified; in real life use freight quotes)
            old_mult = self.ship_mode_cost_multiplier.get(po.ship_mode, 1.0)
            new_mult = self.ship_mode_cost_multiplier.get("air", 1.35)
            s.estimated_cost_delta = (new_mult - old_mult) * (po.qty * po.unit_cost) * 0.10  # 10% freight proxy

            # service impact proxy
            s.estimated_service_level_delta = 0.02 + 0.03 * float(base_stockout >= 0.65)
            s.estimated_late_risk_delta = -0.20
            s.estimated_stockout_risk_delta = -0.15

            if action == POAction.EXPEDITE_AND_INCREASE:
                s.recommended_qty = self._round_qty(po, po.qty * 1.20)  # +20%
                # additional item cost
                s.estimated_cost_delta += (s.recommended_qty - po.qty) * po.unit_cost
                s.estimated_service_level_delta += 0.02
                s.estimated_stockout_risk_delta -= 0.10

        elif action == POAction.SWITCH_SUPPLIER:
            alt = self._find_alternative_supplier(po, suppliers)
            if alt:
                s.recommended_supplier_id = alt[0]
                s.recommended_supplier_name = alt[1]
            s.estimated_cost_delta = (po.qty * po.unit_cost) * 0.02  # 2% switching friction proxy
            s.estimated_service_level_delta = 0.03
            s.estimated_late_risk_delta = -0.18
            s.estimated_stockout_risk_delta = -0.10

        elif action == POAction.SPLIT_ORDER:
            # Split: 30% air (fast), 70% current mode
            q_fast = self._round_qty(po, po.qty * 0.30)
            q_std = self._round_qty(po, po.qty - q_fast)

            now = datetime.utcnow()
            remaining = max(1.0, (po.promised_delivery_date - now).total_seconds() / 86400.0)
            fast_date = now + timedelta(days=float(remaining * self.ship_mode_lead_time_multiplier.get("air", 0.35)))
            std_date = po.promised_delivery_date

            s.split_plan = [
                {"qty": q_fast, "ship_mode": "air", "promised_delivery_date": fast_date.isoformat()},
                {"qty": q_std, "ship_mode": po.ship_mode, "promised_delivery_date": std_date.isoformat()},
            ]

            old_mult = self.ship_mode_cost_multiplier.get(po.ship_mode, 1.0)
            air_mult = self.ship_mode_cost_multiplier.get("air", 1.35)
            # freight proxy for fast portion only
            s.estimated_cost_delta = (air_mult - old_mult) * (q_fast * po.unit_cost) * 0.10
            s.estimated_service_level_delta = 0.025
            s.estimated_late_risk_delta = -0.12
            s.estimated_stockout_risk_delta = -0.10

        elif action == POAction.DEFER:
            s.recommended_promised_delivery_date = po.promised_delivery_date + timedelta(days=14)
            s.estimated_cost_delta = -(po.qty * po.unit_cost) * 0.005  # small carrying-cost relief proxy
            s.estimated_service_level_delta = -0.005
            s.estimated_late_risk_delta = 0.03
            s.estimated_stockout_risk_delta = 0.02

        elif action == POAction.CANCEL:
            s.recommended_qty = 0.0
            s.estimated_cost_delta = -(po.qty * po.unit_cost) * 0.98  # assume most cost avoided
            s.estimated_service_level_delta = -0.02
            s.estimated_late_risk_delta = 0.0
            s.estimated_stockout_risk_delta = 0.08

        elif action == POAction.INCREASE_QTY:
            s.recommended_qty = self._round_qty(po, po.qty * 1.15)
            s.estimated_cost_delta = (s.recommended_qty - po.qty) * po.unit_cost
            s.estimated_service_level_delta = 0.015
            s.estimated_stockout_risk_delta = -0.08

        elif action == POAction.DECREASE_QTY:
            s.recommended_qty = self._round_qty(po, po.qty * 0.85)
            s.estimated_cost_delta = (s.recommended_qty - po.qty) * po.unit_cost
            s.estimated_service_level_delta = -0.01
            s.estimated_stockout_risk_delta = 0.04

        # clamp deltas
        s.estimated_service_level_delta = float(np.clip(s.estimated_service_level_delta, -0.10, 0.10))
        s.estimated_late_risk_delta = float(np.clip(s.estimated_late_risk_delta, -1.0, 1.0))
        s.estimated_stockout_risk_delta = float(np.clip(s.estimated_stockout_risk_delta, -1.0, 1.0))
        return s

    def _round_qty(self, po: PurchaseOrder, qty: float) -> float:
        # respect MOQ if provided
        if po.min_order_qty:
            moq = float(po.min_order_qty)
            qty = max(moq, qty)
        # round to sensible increments
        return float(max(0.0, round(qty)))

    # -------- LLM rationale --------

    async def _attach_llm_rationales(self, company: Company, suggestions: List[POSuggestion]) -> None:
        async def one(s: POSuggestion) -> None:
            try:
                prompt = self._rationale_prompt(company, s)
                resp = await self.model.generate_content_async(prompt)
                s.llm_rationale = (resp.text or "").strip()
            except Exception as e:
                logger.warning(f"LLM rationale failed for PO {s.po_id}: {e}")
                s.llm_rationale = None

        await asyncio.gather(*[one(s) for s in suggestions])

    def _rationale_prompt(self, company: Company, s: POSuggestion) -> str:
        # Keep it short + structured so it’s consistent for an “agent”
        return f"""
You are a supply-chain planning assistant. Write a concise, professional rationale (6-10 bullet points)
for a purchase order adjustment recommendation.

Company:
- Name: {getattr(company, "name", "Unknown")}
- Industry: {getattr(company, "industry", "Unknown")}

Recommendation:
- PO ID: {s.po_id}
- Action: {s.action.value}
- Priority: {s.priority}
- Confidence: {s.confidence:.2f}

Proposed Edits (if any):
- recommended_qty: {s.recommended_qty}
- recommended_ship_mode: {s.recommended_ship_mode}
- recommended_promised_delivery_date: {s.recommended_promised_delivery_date}
- recommended_supplier: {s.recommended_supplier_name}

Estimated Impacts:
- cost_delta: {s.estimated_cost_delta:.2f}
- service_level_delta: {s.estimated_service_level_delta:.3f}
- late_risk_delta: {s.estimated_late_risk_delta:.3f}
- stockout_risk_delta: {s.estimated_stockout_risk_delta:.3f}

Reasons already detected:
- {("; ".join(s.reasons or [])).strip()}

Output requirements:
- Bullet list only
- Mention cost vs service trade-off explicitly
- End with a one-line "Next step" recommendation (e.g., approve + execute, or review by procurement).
""".strip()

    # -------- summary + ERP change mapping --------

    def _summarize(self, suggestions: List[POSuggestion]) -> Dict[str, Any]:
        if not suggestions:
            return {"message": "No adjustments recommended."}

        by_action: Dict[str, int] = {}
        total_cost_delta = 0.0
        avg_service = []
        critical = 0

        for s in suggestions:
            by_action[s.action.value] = by_action.get(s.action.value, 0) + 1
            total_cost_delta += float(s.estimated_cost_delta)
            avg_service.append(float(s.estimated_service_level_delta))
            if s.priority == "critical":
                critical += 1

        return {
            "total_suggestions": len(suggestions),
            "critical_count": critical,
            "by_action": by_action,
            "estimated_total_cost_delta": total_cost_delta,
            "estimated_avg_service_level_delta": float(np.mean(avg_service)) if avg_service else 0.0,
        }

    def _suggestion_to_erp_change(self, s: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert suggestion JSON into an ERP change request payload.
        Your ERPIntegrator.apply_po_changes() can interpret this.
        """
        return {
            "po_id": s["po_id"],
            "action": s["action"],
            "recommended_qty": s.get("recommended_qty"),
            "recommended_promised_delivery_date": s.get("recommended_promised_delivery_date"),
            "recommended_ship_mode": s.get("recommended_ship_mode"),
            "recommended_supplier_id": s.get("recommended_supplier_id"),
            "split_plan": s.get("split_plan"),
            "meta": {
                "confidence": s.get("confidence"),
                "priority": s.get("priority"),
                "reasons": s.get("reasons", []),
            },
        }