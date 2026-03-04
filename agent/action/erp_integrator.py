"""
src/action/erp_integrator.py

Comprehensive (but still mock-friendly) ERP integrator that supports:
- Supplier requirements updates
- Supplier master data updates
- Inventory optimization + inventory target updates
- Logistics alternative route identification + route updates
- Purchase order (PO) adjustment suggestions + applying adjustments
- Integration/audit logging + basic validation rules

This file is designed to "work" end-to-end for your agent even without a real ERP:
it maintains an in-memory mock ERP state and returns structured results your
ActionExecutor can consume.

If you later connect a real ERP (SAP/Oracle/custom), replace _mock_erp_call and
the mock state methods with real API calls, keeping the public method signatures.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import Any, Dict, List, Optional, Tuple

import google.generativeai as genai

from agent.config import settings
from agent.models import Company, Supplier, MitigationAction, RiskAssessment, Disruption

logger = logging.getLogger(__name__)


# -----------------------------
# Helpers / Types
# -----------------------------

@dataclass
class ERPRequestMeta:
    request_id: str
    timestamp: datetime
    module: str
    operation: str


def _now() -> datetime:
    return datetime.utcnow()


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _parse_days(value: Any, default_days: int = 30) -> int:
    """
    Accepts formats like:
      - 30
      - "30_days"
      - "30 days"
      - "2_months" / "2 months"
      - "1_year" / "1 year"
    """
    if isinstance(value, (int, float)):
        return max(0, int(value))

    if not isinstance(value, str):
        return default_days

    s = value.strip().lower()
    # normalize separators
    s = s.replace("-", " ").replace("_", " ")

    parts = s.split()
    if not parts:
        return default_days

    # e.g., "25 days"
    num = _safe_int(parts[0], default_days)

    unit = parts[1] if len(parts) > 1 else "days"
    if unit.startswith("day"):
        return num
    if unit.startswith("month"):
        return num * 30
    if unit.startswith("year"):
        return num * 365

    return default_days


# -----------------------------
# ERP Integrator
# -----------------------------

class ERPIntegrator:
    """
    Comprehensive ERP adapter with mock state.

    Public methods used by your ActionExecutor:
      - update_supplier_requirements(company, action)
      - update_supplier_master_data(company, supplier_data)
      - calculate_optimal_inventory(company, action)
      - update_inventory_levels(company, inventory_recommendations)
      - identify_alternative_routes(company, action)
      - update_logistics_routes(company, route_alternatives)

    Procurement-focused methods for Purchase Order adjustment suggestions:
      - fetch_open_purchase_orders(company, suppliers=None)
      - generate_po_adjustment_suggestions(company, risk_assessment, disruptions, suppliers=None, horizon_days=90)
      - apply_po_adjustments(company, adjustments)

    Optional:
      - get_integration_status(company_id)
      - get_mock_state_snapshot(company_id)
    """

    def __init__(self) -> None:
        # Gemini (optional) — keep agent working even if LLM fails.
        self.llm_enabled = True
        try:
            genai.configure(api_key=settings.google_api_key)
            self.model = genai.GenerativeModel(settings.gemini_model)
        except Exception as e:
            self.llm_enabled = False
            self.model = None
            logger.warning(f"Gemini init failed; continuing without LLM. Error: {e}")

        # Mock ERP systems configurations (placeholders)
        self.erp_systems: Dict[str, Dict[str, Any]] = {
            "sap": {
                "api_endpoint": "https://api.sap.com/mock",
                "authentication": "oauth2",
                "modules": ["procurement", "inventory", "logistics", "supplier_management"],
            },
            "oracle": {
                "api_endpoint": "https://api.oracle.com/mock",
                "authentication": "basic_auth",
                "modules": ["purchasing", "inventory", "supply_chain"],
            },
            "custom": {
                "api_endpoint": "https://api.company-erp.com/mock",
                "authentication": "api_key",
                "modules": ["all"],
            },
        }

        # Integration / audit logs
        self.integration_logs: List[Dict[str, Any]] = []

        # In-memory mock ERP state keyed by company_id
        self._state: Dict[str, Dict[str, Any]] = {}

        # Default business rules (can be overridden per company in future)
        self._rules = {
            "po": {
                "max_qty_increase_pct": 0.50,    # avoid insane jumps without approvals
                "max_qty_decrease_pct": 0.70,    # can cut but not to zero without cancel
                "max_date_push_days": 45,        # max delay shift
                "max_date_pull_days": 30,        # max expedite shift
                "min_line_qty": 1,
                "default_moq": 100,
                "default_lead_time_days": 30,
                "expedite_cost_premium_pct": 0.15,
                "split_order_max_parts": 3,
            },
            "inventory": {
                "carrying_cost_rate": 0.25,      # 25% annual carrying cost
                "service_level_default": 0.95,
            },
        }

    # -----------------------------
    # Mock ERP state initialization
    # -----------------------------

    def _ensure_company_state(self, company: Company) -> None:
        cid = str(company.id)
        if cid in self._state:
            return

        # Mock suppliers map can be enriched later
        self._state[cid] = {
            "suppliers": {},   # supplier_id -> supplier master info
            "inventory": {
                "categories": {
                    "critical_components": {"on_hand": 800, "safety_stock": 500, "reorder_point": 300, "eoq": 800},
                    "standard_components": {"on_hand": 1500, "safety_stock": 1000, "reorder_point": 600, "eoq": 1500},
                    "raw_materials": {"on_hand": 3500, "safety_stock": 2000, "reorder_point": 1200, "eoq": 3000},
                },
                "last_updated": _iso(_now()),
            },
            "logistics": {
                "active_routes": [
                    {"route_id": "BASE_ASIA_NA", "origin": "Asia", "destination": "North America", "mode": "sea", "transit_days": 30, "cost_multiplier": 1.0},
                    {"route_id": "BASE_EU_NA", "origin": "Europe", "destination": "North America", "mode": "sea", "transit_days": 21, "cost_multiplier": 1.0},
                ],
                "last_updated": _iso(_now()),
            },
            "procurement": {
                "purchase_orders": self._seed_mock_purchase_orders(company),
                "last_updated": _iso(_now()),
            },
        }

    def _seed_mock_purchase_orders(self, company: Company) -> List[Dict[str, Any]]:
        """
        Creates a realistic-ish set of open POs to make PO adjustment logic usable.
        """
        base_date = date.today()
        pos: List[Dict[str, Any]] = []

        for i in range(6):
            po_number = f"PO-{base_date.strftime('%Y%m%d')}-{i+1:03d}"
            supplier_id = f"SUP-{(i % 3) + 1:03d}"
            supplier_name = f"Supplier {(i % 3) + 1}"
            po = {
                "po_number": po_number,
                "company_id": str(company.id),
                "status": "open",  # open / partially_received / closed / cancelled
                "supplier_id": supplier_id,
                "supplier_name": supplier_name,
                "created_at": _iso(_now() - timedelta(days=10 + i)),
                "requested_delivery_date": (base_date + timedelta(days=20 + i * 3)).isoformat(),
                "incoterms": "FOB",
                "payment_terms": "NET_30",
                "currency": "USD",
                "lines": [
                    {
                        "line_id": f"{po_number}-L1",
                        "item_code": f"ITEM-{100+i:03d}",
                        "category": "critical_components" if i % 2 == 0 else "standard_components",
                        "quantity": 800 + i * 50,
                        "unit_price": 45.0 + i,
                        "moq": 100,
                        "lead_time_days": 30 + (i % 2) * 7,
                        "priority": "high" if i % 2 == 0 else "medium",
                    },
                    {
                        "line_id": f"{po_number}-L2",
                        "item_code": f"ITEM-{200+i:03d}",
                        "category": "raw_materials",
                        "quantity": 2000 + i * 100,
                        "unit_price": 12.5 + 0.2 * i,
                        "moq": 200,
                        "lead_time_days": 35,
                        "priority": "medium",
                    },
                ],
                "notes": [],
            }
            pos.append(po)

        return pos

    # -----------------------------
    # Public: Supplier Management
    # -----------------------------

    async def update_supplier_requirements(self, company: Company, action: MitigationAction) -> Dict[str, Any]:
        """Update ERP with new supplier requirements (mock)."""
        self._ensure_company_state(company)
        payload = await self._generate_supplier_requirements(company, action)

        meta, resp = await self._mock_erp_call("supplier_management", "update_requirements", payload)
        self._log_integration(company_id=str(company.id), action_id=getattr(action, "id", None),
                              integration_type="supplier_requirements_update", request=payload, response=resp, meta=meta)

        return {
            "integration_type": "supplier_requirements_update",
            "requirements": payload,
            "erp_response": resp,
            "updated_records": resp.get("updated_records", 0),
            "success": resp.get("success", False),
            "timestamp": _now(),
        }

    async def update_supplier_master_data(self, company: Company, supplier_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update supplier master data in ERP (mock)."""
        self._ensure_company_state(company)
        updates = await self._prepare_supplier_updates(supplier_data)

        # Update mock state
        state_suppliers = self._state[str(company.id)]["suppliers"]
        for sup in updates.get("supplier_updates", []):
            state_suppliers[sup["supplier_id"]] = sup

        meta, resp = await self._mock_erp_call("supplier_management", "update_master_data", updates)
        self._log_integration(company_id=str(company.id), action_id=None,
                              integration_type="supplier_master_data_update", request=updates, response=resp, meta=meta)

        return {
            "integration_type": "supplier_master_data_update",
            "updates": updates,
            "erp_response": resp,
            "updated_suppliers": resp.get("updated_records", 0),
            "success": resp.get("success", False),
            "timestamp": _now(),
        }

    # -----------------------------
    # Public: Inventory
    # -----------------------------

    async def calculate_optimal_inventory(self, company: Company, action: MitigationAction) -> Dict[str, Any]:
        """Calculate optimal inventory levels (mock + structured)."""
        self._ensure_company_state(company)
        params = await self._generate_inventory_optimization_params(company, action)
        optimal_levels = await self._mock_inventory_calculation(company, params)

        return {
            "optimization_type": "inventory_optimization",
            "parameters": params,
            "optimal_levels": optimal_levels,
            "recommendations": await self._generate_inventory_recommendations(optimal_levels),
            "calculated_at": _now(),
        }

    async def update_inventory_levels(self, company: Company, inventory_recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """Update ERP with new inventory targets (mock) and update in-memory state."""
        self._ensure_company_state(company)
        updates = await self._prepare_inventory_updates(inventory_recommendations)

        # Update mock state
        inv = self._state[str(company.id)]["inventory"]["categories"]
        for u in updates.get("updates", []):
            cat = u["item_category"]
            if cat in inv:
                inv[cat]["safety_stock"] = _safe_int(u["safety_stock"], inv[cat]["safety_stock"])
                inv[cat]["reorder_point"] = _safe_int(u["reorder_point"], inv[cat]["reorder_point"])
                inv[cat]["eoq"] = _safe_int(u["economic_order_quantity"], inv[cat]["eoq"])
        self._state[str(company.id)]["inventory"]["last_updated"] = _iso(_now())

        meta, resp = await self._mock_erp_call("inventory", "update_levels", updates)
        self._log_integration(company_id=str(company.id), action_id=None,
                              integration_type="inventory_levels_update", request=updates, response=resp, meta=meta)

        return {
            "integration_type": "inventory_levels_update",
            "updates": updates,
            "erp_response": resp,
            "updated_items": resp.get("updated_records", 0),
            "success": resp.get("success", False),
            "timestamp": _now(),
        }

    # -----------------------------
    # Public: Logistics
    # -----------------------------

    async def identify_alternative_routes(self, company: Company, action: MitigationAction) -> Dict[str, Any]:
        """Identify alternative shipping routes (mock)."""
        self._ensure_company_state(company)
        params = await self._generate_route_analysis_params(company, action)
        alternative_routes = await self._mock_route_identification(params)

        return {
            "analysis_type": "alternative_routes",
            "parameters": params,
            "alternative_routes": alternative_routes,
            "recommendations": await self._generate_route_recommendations(alternative_routes),
            "analyzed_at": _now(),
        }

    async def update_logistics_routes(self, company: Company, route_alternatives: Dict[str, Any]) -> Dict[str, Any]:
        """Update ERP with new logistics routes (mock) and update in-memory state."""
        self._ensure_company_state(company)
        updates = await self._prepare_route_updates(route_alternatives)

        # Update mock state: set active_routes from enabled/active routes
        active = []
        for r in updates.get("route_changes", []):
            if r.get("status") == "active":
                active.append({
                    "route_id": r["route_id"],
                    "origin": r["origin"],
                    "destination": r["destination"],
                    "mode": r["transit_mode"],
                    "transit_days": r["transit_time_days"],
                    "cost_multiplier": r["cost_multiplier"],
                })

        if active:
            self._state[str(company.id)]["logistics"]["active_routes"] = active
            self._state[str(company.id)]["logistics"]["last_updated"] = _iso(_now())

        meta, resp = await self._mock_erp_call("logistics", "update_routes", updates)
        self._log_integration(company_id=str(company.id), action_id=None,
                              integration_type="logistics_routes_update", request=updates, response=resp, meta=meta)

        return {
            "integration_type": "logistics_routes_update",
            "updates": updates,
            "erp_response": resp,
            "updated_routes": resp.get("updated_records", 0),
            "success": resp.get("success", False),
            "timestamp": _now(),
        }

    # -----------------------------
    # Public: Procurement / Purchase Orders
    # -----------------------------

    async def fetch_open_purchase_orders(self, company: Company, suppliers: Optional[List[Supplier]] = None) -> List[Dict[str, Any]]:
        """
        Fetch open purchase orders.
        In a real ERP integration, this would query the ERP.
        """
        self._ensure_company_state(company)
        pos = self._state[str(company.id)]["procurement"]["purchase_orders"]
        return [po for po in pos if po.get("status") in ("open", "partially_received")]

    async def generate_po_adjustment_suggestions(
        self,
        company: Company,
        risk_assessment: RiskAssessment,
        disruptions: Optional[List[Disruption]] = None,
        suppliers: Optional[List[Supplier]] = None,
        horizon_days: int = 90,
    ) -> Dict[str, Any]:
        """
        Comprehensive PO adjustment suggestion engine.

        Produces:
        - per-PO, per-line recommendations (qty/date/split/expedite/supplier swap)
        - costs/benefits trade-offs (estimated)
        - risk rationale & confidence
        - optional LLM narrative to make output "professional"
        """
        self._ensure_company_state(company)
        disruptions = disruptions or []

        open_pos = await self.fetch_open_purchase_orders(company, suppliers=suppliers)
        inventory_state = self._state[str(company.id)]["inventory"]["categories"]

        # Basic disruption signals
        disruption_types = {d.disruption_type for d in disruptions if getattr(d, "disruption_type", None)}
        severe = any(getattr(d, "severity_score", 0) >= 0.75 for d in disruptions)

        overall_risk = getattr(risk_assessment, "overall_risk_score", None)
        if overall_risk is None:
            overall_risk = getattr(risk_assessment, "composite_risk_score", 0.6)

        suggestions: List[Dict[str, Any]] = []
        portfolio_summary = {
            "total_open_pos": len(open_pos),
            "total_lines": sum(len(po.get("lines", [])) for po in open_pos),
            "high_priority_lines": 0,
            "estimated_incremental_cost": 0.0,
            "estimated_service_level_uplift": 0.0,
            "estimated_risk_reduction": 0.0,
        }

        for po in open_pos:
            po_suggestions = await self._suggest_adjustments_for_po(
                company=company,
                po=po,
                inventory_state=inventory_state,
                overall_risk=overall_risk,
                disruption_types=disruption_types,
                severe_disruption=severe,
                suppliers=suppliers,
            )
            suggestions.append(po_suggestions)

            # roll-up
            portfolio_summary["high_priority_lines"] += po_suggestions["summary"]["high_priority_lines"]
            portfolio_summary["estimated_incremental_cost"] += po_suggestions["summary"]["estimated_incremental_cost"]
            portfolio_summary["estimated_service_level_uplift"] += po_suggestions["summary"]["estimated_service_level_uplift"]
            portfolio_summary["estimated_risk_reduction"] += po_suggestions["summary"]["estimated_risk_reduction"]

        # normalize some roll-ups
        if portfolio_summary["total_open_pos"] > 0:
            portfolio_summary["estimated_service_level_uplift"] /= portfolio_summary["total_open_pos"]
            portfolio_summary["estimated_risk_reduction"] /= portfolio_summary["total_open_pos"]

        narrative = None
        if self.llm_enabled:
            narrative = await self._llm_generate_po_narrative(company, risk_assessment, disruptions, portfolio_summary, suggestions)

        return {
            "type": "po_adjustment_suggestions",
            "company_id": str(company.id),
            "generated_at": _now(),
            "horizon_days": horizon_days,
            "inputs": {
                "overall_risk_score": overall_risk,
                "disruption_types": sorted(list(disruption_types)),
                "severe_disruption": severe,
                "open_po_count": len(open_pos),
            },
            "portfolio_summary": portfolio_summary,
            "suggestions": suggestions,
            "narrative": narrative,
        }

    async def apply_po_adjustments(self, company: Company, adjustments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply PO adjustments to mock ERP state. In production, this would call ERP APIs.

        Expected adjustments format:
        {
          "adjustments": [
            {
              "po_number": "...",
              "changes": [
                 {"line_id": "...", "type": "qty_change", "new_quantity": 900},
                 {"type": "delivery_date_change", "new_date": "YYYY-MM-DD"},
                 {"type": "supplier_change", "new_supplier_id": "SUP-002"},
                 {"type": "split_order", "line_id": "...", "parts": [...]},
                 {"type": "expedite", "line_id": "...", "premium_pct": 0.15}
              ]
            }, ...
          ]
        }
        """
        self._ensure_company_state(company)
        cid = str(company.id)
        pos = self._state[cid]["procurement"]["purchase_orders"]

        applied: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []

        by_po = {po["po_number"]: po for po in pos}

        for adj in adjustments.get("adjustments", []):
            po_number = adj.get("po_number")
            if po_number not in by_po:
                errors.append({"po_number": po_number, "error": "PO not found"})
                continue

            po = by_po[po_number]
            changes = adj.get("changes", [])
            validate_ok, validate_errs = self._validate_po_changes(po, changes)
            if not validate_ok:
                errors.append({"po_number": po_number, "error": "Validation failed", "details": validate_errs})
                continue

            # Apply changes
            apply_result = self._apply_changes_to_po(po, changes)
            applied.append({"po_number": po_number, "result": apply_result})

        # Simulate ERP call
        meta, resp = await self._mock_erp_call("procurement", "apply_po_adjustments", adjustments)
        self._log_integration(company_id=cid, action_id=None,
                              integration_type="purchase_order_adjustments", request=adjustments, response=resp, meta=meta)

        return {
            "integration_type": "purchase_order_adjustments",
            "success": resp.get("success", False) and not errors,
            "applied": applied,
            "errors": errors,
            "erp_response": resp,
            "timestamp": _now(),
        }

    # -----------------------------
    # Status / Debug helpers
    # -----------------------------

    async def get_integration_status(self, company_id: str) -> Dict[str, Any]:
        """Get integration status and summary stats for a company."""
        company_logs = [log for log in self.integration_logs if log.get("company_id") == company_id]

        total_integrations = len(company_logs)
        successful_integrations = len([log for log in company_logs if log.get("status") == "success"])
        failed_integrations = len([log for log in company_logs if log.get("status") == "error"])

        by_type: Dict[str, Dict[str, int]] = {}
        for log in company_logs:
            t = log.get("integration_type", "unknown")
            by_type.setdefault(t, {"total": 0, "success": 0, "error": 0})
            by_type[t]["total"] += 1
            by_type[t][log.get("status", "success")] += 1

        return {
            "company_id": company_id,
            "total_integrations": total_integrations,
            "successful_integrations": successful_integrations,
            "failed_integrations": failed_integrations,
            "success_rate": (successful_integrations / total_integrations) * 100 if total_integrations > 0 else 0.0,
            "integration_types": by_type,
            "last_integration": company_logs[-1]["timestamp"] if company_logs else None,
            "recent_logs": company_logs[-10:],
        }

    def get_mock_state_snapshot(self, company_id: str) -> Dict[str, Any]:
        """Return current mock ERP state snapshot (useful for debugging/demo)."""
        return self._state.get(company_id, {})

    # -----------------------------
    # Internal: Suggestion logic
    # -----------------------------

    async def _suggest_adjustments_for_po(
        self,
        company: Company,
        po: Dict[str, Any],
        inventory_state: Dict[str, Dict[str, Any]],
        overall_risk: float,
        disruption_types: set,
        severe_disruption: bool,
        suppliers: Optional[List[Supplier]] = None,
    ) -> Dict[str, Any]:
        """
        Generate structured suggestions for a single PO.
        """
        po_number = po["po_number"]
        req_date = po.get("requested_delivery_date")
        lines = po.get("lines", [])

        per_line: List[Dict[str, Any]] = []
        summary = {
            "po_number": po_number,
            "high_priority_lines": 0,
            "estimated_incremental_cost": 0.0,
            "estimated_service_level_uplift": 0.0,
            "estimated_risk_reduction": 0.0,
        }

        # risk multipliers
        shipping_risk = 0.12 if "shipping" in disruption_types else 0.0
        supplier_risk = 0.15 if "supplier" in disruption_types else 0.0
        geo_risk = 0.10 if "geopolitical" in disruption_types else 0.0
        climate_risk = 0.08 if "climate" in disruption_types else 0.0
        shock = shipping_risk + supplier_risk + geo_risk + climate_risk
        if severe_disruption:
            shock *= 1.5

        for line in lines:
            priority = line.get("priority", "medium")
            category = line.get("category", "standard_components")
            qty = _safe_int(line.get("quantity"), 0)
            moq = _safe_int(line.get("moq"), self._rules["po"]["default_moq"])
            lead_time_days = _safe_int(line.get("lead_time_days"), self._rules["po"]["default_lead_time_days"])
            unit_price = _safe_float(line.get("unit_price"), 0.0)

            if priority == "high":
                summary["high_priority_lines"] += 1

            inv = inventory_state.get(category, {})
            on_hand = _safe_int(inv.get("on_hand"), 0)
            safety = _safe_int(inv.get("safety_stock"), 0)
            reorder_point = _safe_int(inv.get("reorder_point"), 0)

            # Simple “inventory pressure” signal:
            # - if on_hand < reorder_point => risk of stockout, consider expedite or increase qty
            # - if on_hand >> safety_stock and risk low => consider delaying to reduce holding cost
            stockout_risk = 0.0
            if on_hand < reorder_point:
                stockout_risk = 0.7
            elif on_hand < safety:
                stockout_risk = 0.4
            else:
                stockout_risk = 0.2

            # Combine risk signals
            line_risk_score = min(1.0, 0.45 * overall_risk + 0.35 * stockout_risk + 0.20 * shock)

            actions: List[Dict[str, Any]] = []

            # 1) Expedite if high priority and risk elevated
            if (priority == "high" and line_risk_score >= 0.65) or (stockout_risk >= 0.7 and line_risk_score >= 0.55):
                pull_days = min(self._rules["po"]["max_date_pull_days"], max(7, lead_time_days // 3))
                premium = self._rules["po"]["expedite_cost_premium_pct"]
                incremental_cost = qty * unit_price * premium
                actions.append({
                    "type": "expedite",
                    "line_id": line["line_id"],
                    "recommended_pull_in_days": pull_days,
                    "estimated_cost_increase": round(incremental_cost, 2),
                    "confidence": round(min(0.95, 0.60 + line_risk_score * 0.35), 2),
                    "rationale": "High risk of service impact; expediting reduces lead-time exposure.",
                })
                summary["estimated_incremental_cost"] += incremental_cost
                summary["estimated_service_level_uplift"] += 0.06
                summary["estimated_risk_reduction"] += 0.05

            # 2) Increase qty (buffer) if stockout risk is high
            if stockout_risk >= 0.7:
                inc_pct = min(self._rules["po"]["max_qty_increase_pct"], 0.25 + 0.15 * overall_risk)
                new_qty = int(qty * (1.0 + inc_pct))
                new_qty = max(new_qty, moq)
                delta = new_qty - qty
                if delta >= self._rules["po"]["min_line_qty"]:
                    actions.append({
                        "type": "qty_change",
                        "line_id": line["line_id"],
                        "current_quantity": qty,
                        "new_quantity": new_qty,
                        "estimated_cost_increase": round(delta * unit_price, 2),
                        "confidence": round(min(0.9, 0.55 + stockout_risk * 0.45), 2),
                        "rationale": "Inventory below reorder point; increasing PO quantity reduces stockout probability.",
                    })
                    summary["estimated_incremental_cost"] += delta * unit_price
                    summary["estimated_service_level_uplift"] += 0.05
                    summary["estimated_risk_reduction"] += 0.04

            # 3) Delay / push date if risk low and inventory comfortable (reduce carrying cost)
            if stockout_risk <= 0.2 and overall_risk <= 0.5 and priority != "high":
                push_days = min(self._rules["po"]["max_date_push_days"], 14)
                actions.append({
                    "type": "delivery_date_push",
                    "po_number": po_number,
                    "recommended_push_out_days": push_days,
                    "estimated_carrying_cost_savings": round(qty * unit_price * 0.02, 2),
                    "confidence": 0.65,
                    "rationale": "Inventory is healthy; deferring delivery can reduce carrying costs without harming service.",
                })
                summary["estimated_service_level_uplift"] += 0.01
                summary["estimated_risk_reduction"] += 0.01

            # 4) Split order if disruption shock is high (reduce single-point-of-failure)
            if shock >= 0.18 and qty >= 2 * moq:
                parts = min(self._rules["po"]["split_order_max_parts"], 2 + int(shock * 2))
                split_qty = max(moq, qty // parts)
                plan = [{"part": p + 1, "quantity": split_qty} for p in range(parts)]
                remainder = qty - split_qty * parts
                if remainder > 0:
                    plan[-1]["quantity"] += remainder
                actions.append({
                    "type": "split_order",
                    "line_id": line["line_id"],
                    "parts": plan,
                    "confidence": round(min(0.9, 0.55 + shock * 1.5), 2),
                    "rationale": "Elevated disruption risk; splitting reduces exposure and improves scheduling flexibility.",
                })
                summary["estimated_risk_reduction"] += 0.05
                summary["estimated_service_level_uplift"] += 0.03

            # 5) Supplier reallocation suggestion (only as suggestion; actual optimization is in Planning Engine)
            # If overall risk is high or supplier disruptions present, propose alternate supplier placeholder.
            if overall_risk >= 0.7 or "supplier" in disruption_types:
                # choose a "candidate" supplier from provided suppliers if available
                candidate = None
                if suppliers:
                    # Prefer suppliers with lower risk_score if present
                    suppliers_sorted = sorted(
                        suppliers,
                        key=lambda s: getattr(s, "risk_score", 0.5)
                    )
                    if suppliers_sorted:
                        candidate = suppliers_sorted[0]

                actions.append({
                    "type": "supplier_swap_suggestion",
                    "line_id": line["line_id"],
                    "current_supplier_id": po.get("supplier_id"),
                    "suggested_supplier_id": getattr(candidate, "id", "ALT-SUP-001") if candidate else "ALT-SUP-001",
                    "confidence": round(min(0.85, 0.50 + overall_risk * 0.45), 2),
                    "rationale": "Supplier disruption risk is elevated; reallocating spend can reduce concentration and outage exposure.",
                })
                summary["estimated_risk_reduction"] += 0.03

            per_line.append({
                "line_id": line["line_id"],
                "item_code": line.get("item_code"),
                "category": category,
                "priority": priority,
                "computed_signals": {
                    "stockout_risk": round(stockout_risk, 2),
                    "shock": round(shock, 2),
                    "line_risk_score": round(line_risk_score, 2),
                },
                "recommended_actions": actions,
            })

        # Normalize summary (cap)
        summary["estimated_service_level_uplift"] = float(min(0.25, summary["estimated_service_level_uplift"]))
        summary["estimated_risk_reduction"] = float(min(0.30, summary["estimated_risk_reduction"]))

        return {
            "po_number": po_number,
            "supplier_id": po.get("supplier_id"),
            "supplier_name": po.get("supplier_name"),
            "requested_delivery_date": req_date,
            "line_suggestions": per_line,
            "summary": summary,
        }

    def _validate_po_changes(self, po: Dict[str, Any], changes: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        errs: List[str] = []
        lines_by_id = {l["line_id"]: l for l in po.get("lines", [])}

        for ch in changes:
            ctype = ch.get("type")
            line_id = ch.get("line_id")

            if ctype in ("qty_change", "expedite", "split_order") and line_id not in lines_by_id:
                errs.append(f"Line not found for change: {ctype} line_id={line_id}")
                continue

            if ctype == "qty_change":
                line = lines_by_id[line_id]
                cur = _safe_int(line.get("quantity"), 0)
                newq = _safe_int(ch.get("new_quantity"), -1)
                moq = _safe_int(line.get("moq"), self._rules["po"]["default_moq"])
                if newq < moq:
                    errs.append(f"qty_change below MOQ for {line_id}: new={newq}, moq={moq}")
                if cur > 0:
                    inc_pct = (newq - cur) / cur
                    if inc_pct > self._rules["po"]["max_qty_increase_pct"]:
                        errs.append(f"qty_change increase too large for {line_id}: {inc_pct:.2%}")
                    if inc_pct < -self._rules["po"]["max_qty_decrease_pct"]:
                        errs.append(f"qty_change decrease too large for {line_id}: {inc_pct:.2%}")

            if ctype in ("delivery_date_change", "delivery_date_push", "delivery_date_pull"):
                nd = ch.get("new_date")
                if not isinstance(nd, str) or len(nd) < 8:
                    errs.append("Invalid new_date for delivery_date change")

            if ctype == "split_order":
                parts = ch.get("parts", [])
                if not isinstance(parts, list) or len(parts) < 2:
                    errs.append(f"split_order must have >=2 parts for {line_id}")
                if len(parts) > self._rules["po"]["split_order_max_parts"]:
                    errs.append(f"split_order too many parts for {line_id}")
                total = sum(_safe_int(p.get("quantity"), 0) for p in parts)
                cur = _safe_int(lines_by_id[line_id].get("quantity"), 0)
                if total != cur:
                    errs.append(f"split_order quantities must sum to current qty for {line_id}: total={total}, current={cur}")

        return (len(errs) == 0), errs

    def _apply_changes_to_po(self, po: Dict[str, Any], changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Mutates the PO dict in-place; returns applied changes summary.
        """
        lines_by_id = {l["line_id"]: l for l in po.get("lines", [])}
        applied = []
        for ch in changes:
            ctype = ch.get("type")

            if ctype == "qty_change":
                line = lines_by_id[ch["line_id"]]
                old = line["quantity"]
                line["quantity"] = int(ch["new_quantity"])
                applied.append({"type": ctype, "line_id": ch["line_id"], "old": old, "new": line["quantity"]})

            elif ctype in ("delivery_date_change", "delivery_date_push", "delivery_date_pull"):
                old = po.get("requested_delivery_date")
                po["requested_delivery_date"] = ch.get("new_date", old)
                applied.append({"type": ctype, "old": old, "new": po["requested_delivery_date"]})

            elif ctype == "supplier_change":
                old = po.get("supplier_id")
                po["supplier_id"] = ch.get("new_supplier_id", old)
                po["supplier_name"] = ch.get("new_supplier_name", po.get("supplier_name"))
                applied.append({"type": ctype, "old": old, "new": po.get("supplier_id")})

            elif ctype == "expedite":
                line = lines_by_id[ch["line_id"]]
                premium = _safe_float(ch.get("premium_pct"), self._rules["po"]["expedite_cost_premium_pct"])
                po.setdefault("notes", []).append(
                    f"Expedite requested for line {ch['line_id']} (premium {premium:.0%}) at {_iso(_now())}"
                )
                line["unit_price"] = round(_safe_float(line["unit_price"]) * (1.0 + premium), 4)
                applied.append({"type": ctype, "line_id": ch["line_id"], "premium_pct": premium})

            elif ctype == "split_order":
                # Represent split as note + metadata; real ERP would create child POs.
                po.setdefault("notes", []).append(
                    f"Split order planned for line {ch['line_id']}: {json.dumps(ch.get('parts', []))}"
                )
                applied.append({"type": ctype, "line_id": ch["line_id"], "parts": ch.get("parts", [])})

            else:
                applied.append({"type": ctype, "status": "ignored_or_unhandled", "details": ch})

        po.setdefault("notes", []).append(f"PO adjusted by autonomous agent at {_iso(_now())}")
        return {"applied_changes": applied, "po_number": po.get("po_number")}

    # -----------------------------
    # LLM narrative (optional)
    # -----------------------------

    async def _llm_generate_po_narrative(
        self,
        company: Company,
        risk_assessment: RiskAssessment,
        disruptions: List[Disruption],
        portfolio_summary: Dict[str, Any],
        suggestions: List[Dict[str, Any]],
    ) -> Optional[str]:
        """
        Generates a professional narrative summary using Gemini.
        If Gemini fails, returns None (agent still works).
        """
        if not self.llm_enabled or self.model is None:
            return None

        # Keep prompt small-ish: summarize counts and top findings only.
        top_risks = []
        for d in disruptions[:5]:
            top_risks.append({
                "type": getattr(d, "disruption_type", "unknown"),
                "severity": getattr(d, "severity_score", None),
                "regions": getattr(d, "affected_regions", None),
            })

        overall_risk = getattr(risk_assessment, "overall_risk_score", None)
        if overall_risk is None:
            overall_risk = getattr(risk_assessment, "composite_risk_score", 0.6)

        # Extract a few “headline” actions
        headline_actions = []
        for po_s in suggestions[:3]:
            for ls in po_s.get("line_suggestions", [])[:2]:
                for act in ls.get("recommended_actions", [])[:1]:
                    headline_actions.append({
                        "po": po_s.get("po_number"),
                        "line": ls.get("line_id"),
                        "action": act.get("type"),
                        "rationale": act.get("rationale"),
                    })
            if len(headline_actions) >= 6:
                break

        prompt = f"""
You are an operations analyst writing a concise executive summary.
Write 1 short paragraph + 4 bullet points.

Company: {company.name} ({company.industry})
Overall risk score: {overall_risk}

Disruption signals (top): {json.dumps(top_risks)}

Portfolio summary: {json.dumps(portfolio_summary)}

Headline recommended actions: {json.dumps(headline_actions)}

Requirements:
- Sound professional and action-oriented.
- Mention cost vs service trade-off briefly.
- Include a clear "next 7 days" action suggestion.
"""

        try:
            resp = await self.model.generate_content_async(prompt)
            return resp.text.strip()
        except Exception as e:
            logger.warning(f"LLM narrative generation failed: {e}")
            return None

    # -----------------------------
    # Internal: Integration logging
    # -----------------------------

    def _log_integration(
        self,
        company_id: str,
        action_id: Optional[Any],
        integration_type: str,
        request: Dict[str, Any],
        response: Dict[str, Any],
        meta: ERPRequestMeta,
    ) -> None:
        status = "success" if response.get("success") else "error"
        entry = {
            "timestamp": meta.timestamp,
            "company_id": company_id,
            "action_id": action_id,
            "integration_type": integration_type,
            "module": meta.module,
            "operation": meta.operation,
            "request_id": meta.request_id,
            "request": request,
            "response": response,
            "status": status,
        }
        self.integration_logs.append(entry)
        logger.info(f"ERP integration log: {integration_type} status={status} req_id={meta.request_id}")

    # -----------------------------
    # Internal: Mock ERP "API"
    # -----------------------------

    async def _mock_erp_call(self, module: str, operation: str, data: Dict[str, Any]) -> Tuple[ERPRequestMeta, Dict[str, Any]]:
        """
        Simulate an ERP call with latency and a consistent response envelope.
        """
        await asyncio.sleep(0.08)

        request_id = f"REQ_{uuid.uuid4().hex[:10].upper()}"
        meta = ERPRequestMeta(
            request_id=request_id,
            timestamp=_now(),
            module=module,
            operation=operation,
        )

        # Better "updated_records" counting across different payload structures
        updated_records = 1
        if isinstance(data, dict):
            if "updates" in data and isinstance(data["updates"], list):
                updated_records = len(data["updates"])
            elif "route_changes" in data and isinstance(data["route_changes"], list):
                updated_records = len(data["route_changes"])
            elif "purchase_orders" in data and isinstance(data["purchase_orders"], list):
                updated_records = len(data["purchase_orders"])
            elif "supplier_updates" in data and isinstance(data["supplier_updates"], list):
                updated_records = len(data["supplier_updates"])
            elif "adjustments" in data and isinstance(data["adjustments"], list):
                updated_records = len(data["adjustments"])

        response = {
            "success": True,
            "message": f"{operation} completed successfully in {module}",
            "timestamp": meta.timestamp.isoformat(),
            "request_id": meta.request_id,
            "updated_records": updated_records,
            "processing_time_ms": 150,
        }
        return meta, response

    # -----------------------------
    # Internal: Supplier requirement payload
    # -----------------------------

    async def _generate_supplier_requirements(self, company: Company, action: MitigationAction) -> Dict[str, Any]:
        return {
            "company_id": str(company.id),
            "action_id": getattr(action, "id", None),
            "requirement_type": "diversification",
            "criteria": {
                "quality_standards": "ISO_9001",
                "delivery_performance": ">95%",
                "geographic_diversification": True,
                "capacity_requirements": "1000_units_month",
                "certification_requirements": ["ISO_14001", "OHSAS_18001"],
            },
            "risk_tolerance": getattr(company, "risk_appetite", 0.5),
            "priority_level": getattr(action, "priority_level", "medium"),
            "effective_date": _now().date().isoformat(),
            "review_date": (_now() + timedelta(days=90)).date().isoformat(),
        }

    async def _prepare_supplier_updates(self, supplier_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert arbitrary supplier input into ERP-friendly master updates.
        """
        updates = supplier_data.get("supplier_updates")
        if isinstance(updates, list) and updates:
            # assume already in correct format
            return {"supplier_updates": updates, "update_reason": supplier_data.get("update_reason", "supplier_update"),
                    "effective_date": _now().date().isoformat()}

        # Otherwise, create a small mock set.
        return {
            "supplier_updates": [
                {
                    "supplier_id": f"SUP-{i+1:03d}",
                    "status": "active",
                    "certification_level": "preferred",
                    "performance_rating": "A",
                    "risk_score": 0.3 + i * 0.1,
                    "diversification_category": "strategic",
                    "contract_terms": {
                        "payment_terms": "NET_45",
                        "delivery_terms": "EXW",
                        "quality_requirements": "ISO_9001",
                        "lead_time_days": 30 + i * 5,
                    },
                }
                for i in range(3)
            ],
            "update_reason": supplier_data.get("update_reason", "Supplier diversification initiative"),
            "effective_date": _now().date().isoformat(),
        }

    # -----------------------------
    # Internal: Inventory methods
    # -----------------------------

    async def _generate_inventory_optimization_params(self, company: Company, action: MitigationAction) -> Dict[str, Any]:
        current_buffer = getattr(company, "inventory_buffer_policy", 0.3)
        target_buffer = min(0.6, current_buffer + 0.2)
        return {
            "company_id": str(company.id),
            "action_id": getattr(action, "id", None),
            "current_buffer_policy": current_buffer,
            "target_buffer_policy": target_buffer,
            "lead_time_sensitivity": getattr(company, "lead_time_sensitivity", 0.5),
            "service_level_target": getattr(company, "service_level_target", self._rules["inventory"]["service_level_default"]),
            "optimization_objectives": ["minimize_stockouts", "optimize_carrying_costs", "improve_service_levels"],
            "constraints": {
                "warehouse_capacity_units": 10000,
                "budget_limit": getattr(action, "estimated_cost", 100000),
                "supplier_lead_time_days": 30,
            },
        }

    async def _mock_inventory_calculation(self, company: Company, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Produces a more internally consistent set of targets based on target buffer.
        """
        buffer = _safe_float(params.get("target_buffer_policy"), 0.4)

        # Scale safety stock & reorder points based on buffer
        base = self._state[str(company.id)]["inventory"]["categories"]

        def scaled(cat: str, base_key: str, factor: float) -> int:
            return int(_safe_int(base[cat][base_key]) * factor)

        # Buffer multiplier: from 0.2..0.6 -> 0.8..1.3 roughly
        mult = 0.7 + buffer

        optimal = {
            "optimal_safety_stock": {
                "critical_components": f"{scaled('critical_components','safety_stock', mult)}_units",
                "standard_components": f"{scaled('standard_components','safety_stock', mult)}_units",
                "raw_materials": f"{scaled('raw_materials','safety_stock', mult)}_units",
            },
            "reorder_points": {
                "critical_components": f"{scaled('critical_components','reorder_point', mult)}_units",
                "standard_components": f"{scaled('standard_components','reorder_point', mult)}_units",
                "raw_materials": f"{scaled('raw_materials','reorder_point', mult)}_units",
            },
            "economic_order_quantities": {
                "critical_components": f"{scaled('critical_components','eoq', 1.0)}_units",
                "standard_components": f"{scaled('standard_components','eoq', 1.0)}_units",
                "raw_materials": f"{scaled('raw_materials','eoq', 1.0)}_units",
            },
            "total_investment_required": params["constraints"]["budget_limit"],
            "expected_service_level_improvement": f"{int(3 + buffer * 10)}%",
            "carrying_cost_increase": f"{int(8 + buffer * 20)}%",
        }
        return optimal

    async def _generate_inventory_recommendations(self, optimal_levels: Dict[str, Any]) -> List[str]:
        return [
            f"Increase safety stock for critical components to {optimal_levels['optimal_safety_stock']['critical_components']}",
            f"Adjust reorder points for standard components to {optimal_levels['reorder_points']['standard_components']}",
            "Implement automated inventory monitoring alerts (stockout + overstock thresholds).",
            "Review supplier lead times quarterly and update reorder points accordingly.",
        ]

    async def _prepare_inventory_updates(self, recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert calculate_optimal_inventory output to ERP update format.
        """
        levels = recommendations.get("optimal_levels", {})
        ss = levels.get("optimal_safety_stock", {})
        rp = levels.get("reorder_points", {})
        eoq = levels.get("economic_order_quantities", {})

        def to_int_units(v: Any, fallback: int) -> int:
            if isinstance(v, str) and v.endswith("_units"):
                return _safe_int(v.replace("_units", ""), fallback)
            return _safe_int(v, fallback)

        return {
            "update_type": "inventory_levels",
            "updates": [
                {
                    "item_category": "critical_components",
                    "safety_stock": to_int_units(ss.get("critical_components"), 500),
                    "reorder_point": to_int_units(rp.get("critical_components"), 300),
                    "economic_order_quantity": to_int_units(eoq.get("critical_components"), 800),
                },
                {
                    "item_category": "standard_components",
                    "safety_stock": to_int_units(ss.get("standard_components"), 1000),
                    "reorder_point": to_int_units(rp.get("standard_components"), 600),
                    "economic_order_quantity": to_int_units(eoq.get("standard_components"), 1500),
                },
                {
                    "item_category": "raw_materials",
                    "safety_stock": to_int_units(ss.get("raw_materials"), 2000),
                    "reorder_point": to_int_units(rp.get("raw_materials"), 1200),
                    "economic_order_quantity": to_int_units(eoq.get("raw_materials"), 3000),
                },
            ],
            "effective_date": _now().date().isoformat(),
            "update_reason": "Supply chain resilience inventory optimization",
        }

    # -----------------------------
    # Internal: Logistics methods
    # -----------------------------

    async def _generate_route_analysis_params(self, company: Company, action: MitigationAction) -> Dict[str, Any]:
        return {
            "company_id": str(company.id),
            "action_id": getattr(action, "id", None),
            "current_routes": self._state[str(company.id)]["logistics"]["active_routes"],
            "risk_factors": ["geopolitical_instability", "port_congestion", "weather_disruptions", "capacity_constraints"],
            "optimization_criteria": ["minimize_transit_time", "reduce_risk_exposure", "control_costs", "improve_reliability"],
            "constraints": {
                "budget_limit": getattr(action, "estimated_cost", 200000),
                "service_level_requirements": "95%",
                "carrier_preferences": ["existing_partners"],
            },
        }

    async def _mock_route_identification(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "alternative_routes": [
                {
                    "route_id": "ALT_001",
                    "origin": "Southeast Asia",
                    "destination": "North America",
                    "transit_mode": "sea_air_combination",
                    "transit_time": "25_days",
                    "cost_increase": "15%",
                    "risk_reduction": "40%",
                    "reliability_score": "0.90",
                },
                {
                    "route_id": "ALT_002",
                    "origin": "South America",
                    "destination": "North America",
                    "transit_mode": "land_transport",
                    "transit_time": "7_days",
                    "cost_increase": "25%",
                    "risk_reduction": "60%",
                    "reliability_score": "0.85",
                },
                {
                    "route_id": "ALT_003",
                    "origin": "Middle East",
                    "destination": "North America",
                    "transit_mode": "sea_transport",
                    "transit_time": "35_days",
                    "cost_increase": "5%",
                    "risk_reduction": "20%",
                    "reliability_score": "0.80",
                },
            ],
            "recommended_routes": ["ALT_001", "ALT_002"],
            "implementation_timeline": "14_days",
            "expected_benefits": {"risk_reduction": "35%", "service_improvement": "10%", "cost_impact": "15%_increase"},
        }

    async def _generate_route_recommendations(self, route_bundle: Dict[str, Any]) -> List[str]:
        recs = []
        alts = route_bundle.get("alternative_routes", [])
        recommended = set(route_bundle.get("recommended_routes", []))
        for r in alts:
            if r.get("route_id") in recommended:
                recs.append(
                    f"Activate {r['route_id']} ({r['origin']}→{r['destination']} via {r['transit_mode']}); "
                    f"risk reduction {r['risk_reduction']} with cost increase {r['cost_increase']}."
                )
        recs += [
            "Monitor route KPIs weekly (transit time, delay variance, cost variance).",
            "Maintain at least one standby route per lane for rapid switching.",
            "Negotiate capacity reservations with carriers for peak periods.",
        ]
        return recs

    async def _prepare_route_updates(self, route_alternatives: Dict[str, Any]) -> Dict[str, Any]:
        alts = route_alternatives.get("alternative_routes", {}).get("alternative_routes", [])
        if not alts:
            # allow passing raw bundle directly too
            alts = route_alternatives.get("alternative_routes", []) or []

        recommended = route_alternatives.get("alternative_routes", {}).get("recommended_routes")
        if recommended is None:
            recommended = route_alternatives.get("recommended_routes", [])
        recommended_set = set(recommended or [])

        def pct_to_mult(pct_str: str, default_mult: float = 1.0) -> float:
            if not isinstance(pct_str, str) or "%" not in pct_str:
                return default_mult
            v = _safe_float(pct_str.replace("%", ""), 0.0) / 100.0
            return round(1.0 + v, 4)

        route_changes = []
        for r in alts:
            transit_days = _parse_days(r.get("transit_time", "30_days"), 30)
            cost_mult = pct_to_mult(r.get("cost_increase", "0%"), 1.0)

            route_changes.append({
                "route_id": r.get("route_id"),
                "status": "active" if r.get("route_id") in recommended_set else "standby",
                "origin": r.get("origin"),
                "destination": r.get("destination"),
                "transit_mode": r.get("transit_mode"),
                "transit_time_days": transit_days,
                "cost_multiplier": cost_mult,
                "priority": "high" if r.get("route_id") in recommended_set else "medium",
            })

        return {
            "update_type": "logistics_routes",
            "route_changes": route_changes,
            "effective_date": _now().date().isoformat(),
            "update_reason": "Supply chain resilience route diversification",
        }