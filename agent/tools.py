import json
import os
import hashlib
import re
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

MEMORY_PATH = os.path.join(os.path.dirname(__file__), "memory.json")
PROFILES_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "company_profiles.json")
DISRUPTIONS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "disruption_signals.json")
LIVE_DISRUPTIONS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "live_disruption_signals.json")
CRITICAL_DISRUPTIONS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "disruption_signals_critical.json")
PIPELINE_CACHE_PATH = os.path.join(os.path.dirname(__file__), "pipeline_cache.json")
EVENT_STATE_PATH = os.path.join(os.path.dirname(__file__), "event_state.json")
WORKFLOW_LOG_PATH = os.path.join(os.path.dirname(__file__), "workflow_execution_log.json")


def _load_memory() -> Dict[str, Any]:
    if not os.path.exists(MEMORY_PATH):
        return {"events": []}
    with open(MEMORY_PATH, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        return {"events": []}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Recover gracefully if memory file gets partially corrupted.
        return {"events": []}


def _load_json(path: str, default: Dict[str, Any]) -> Dict[str, Any]:
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default


def _save_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _save_memory(mem: Dict[str, Any]) -> None:
    _save_json(MEMORY_PATH, mem)


def _load_company_profiles() -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(PROFILES_PATH):
        return {
            "de_semiconductor_auto": {
                "company_id": "de_semiconductor_auto",
                "company_name": "MidMarket Auto Parts Co",
                "region": "Germany",
                "industry": "automotive_parts",
                "risk_appetite": "medium",
                "critical_components": ["semiconductors", "wire_harness"],
                "supplier_concentration": {
                    "semiconductors": {"top_supplier_share": 0.72, "region": "East Asia"},
                    "wire_harness": {"top_supplier_share": 0.55, "region": "Eastern Europe"},
                },
                "inventory_policy": {
                    "semiconductors_days_buffer": 10,
                    "wire_harness_days_buffer": 15,
                },
                "contract_structures": {
                    "semiconductors": "fixed_volume_quarterly",
                    "wire_harness": "rolling_monthly",
                },
                "sla": {"on_time_delivery_target": 0.95, "penalty_per_day_delay_usd": 25000},
                "lead_time_sensitivity": "high",
            }
        }
    with open(PROFILES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("profiles", {})


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def get_company_profile(company_id: str = "de_semiconductor_auto") -> Dict[str, Any]:
    """Hyper-personalized profile by company_id."""
    profiles = _load_company_profiles()
    if company_id in profiles:
        return profiles[company_id]
    return next(iter(profiles.values()))


def _load_signal_templates() -> List[Dict[str, Any]]:
    default_signals = [
        {
            "id": "evt-shipping-redsea",
            "type": "shipping_disruption",
            "region": "Red Sea",
            "severity": 0.82,
            "confidence": 0.75,
            "summary": "Carriers diverting routes causing 10–20 day delays.",
            "affected": ["ocean_freight", "semiconductors"],
            "category": "logistics",
        },
        {
            "id": "evt-geopolitical-export",
            "type": "geopolitical_change",
            "region": "East Asia",
            "severity": 0.66,
            "confidence": 0.71,
            "summary": "New export control checks may delay semiconductor customs clearance.",
            "affected": ["semiconductors", "customs"],
            "category": "procurement",
        },
        {
            "id": "evt-climate-port",
            "type": "climate_event",
            "region": "North Europe",
            "severity": 0.49,
            "confidence": 0.64,
            "summary": "Severe storms create intermittent port closure risk this week.",
            "affected": ["ocean_freight", "finished_goods"],
            "category": "logistics",
        }
    ]

    def _read_signals(path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data.get("signals", [])
        if isinstance(data, list):
            return data
        return []

    source_mode = os.getenv("APP_SIGNAL_SOURCE", "mock").strip().lower()
    signal_profile = os.getenv("APP_SIGNAL_PROFILE", "default").strip().lower()
    mock_path = CRITICAL_DISRUPTIONS_PATH if signal_profile == "critical" else DISRUPTIONS_PATH
    mock_signals = _read_signals(mock_path) or default_signals
    live_signals = _read_signals(LIVE_DISRUPTIONS_PATH)

    if source_mode == "live":
        return live_signals or mock_signals
    if source_mode == "hybrid":
        return (live_signals + mock_signals) if live_signals else mock_signals
    return mock_signals


def _normalize_summary(text: str, max_words: int = 35) -> str:
    words = (text or "").replace("\n", " ").split()
    return " ".join(words[:max_words])


def _rule_relevance(signal: Dict[str, Any], company: Dict[str, Any]) -> float:
    keywords = {"semiconductor", "shipping", "flood", "geopolitical", "insolvency", "port", "storm"}
    summary = str(signal.get("summary", "")).lower()
    hit_count = sum(1 for kw in keywords if kw in summary)
    sev = float(signal.get("severity", 0.0))
    conf = float(signal.get("confidence", 0.0))
    critical = set(company.get("critical_components", []))
    affected = set(signal.get("affected", []))
    component_hit = 1.0 if critical.intersection(affected) else 0.5
    return min(1.0, 0.35 * sev + 0.25 * conf + 0.25 * component_hit + 0.15 * min(1.0, hit_count / 2))


def _classify_signal(signal: Dict[str, Any]) -> Dict[str, Any]:
    signal_type = str(signal.get("type", "")).lower()
    summary = str(signal.get("summary", "")).lower()
    category = str(signal.get("category", "")).lower()
    tokens = set(re.findall(r"\b[a-z0-9_-]+\b", summary))
    if "shipping" in signal_type or "port" in tokens or "ocean" in tokens:
        disruption_type = "shipping_disruption"
    elif "semiconductor" in summary or "chip" in summary:
        disruption_type = "semiconductor_shortage"
    elif "climate" in signal_type or "storm" in summary or "flood" in summary:
        disruption_type = "climate_event"
    elif "geopolitical" in signal_type or "export control" in summary:
        disruption_type = "geopolitical_issue"
    elif "financial" in signal_type or "insolvency" in summary or "debt" in summary:
        disruption_type = "supplier_insolvency"
    elif category == "procurement":
        disruption_type = "procurement_risk"
    else:
        disruption_type = "other"
    return {"disruption_type": disruption_type, "classifier": "lightweight_rules_v1"}


def _tier_from_severity(severity: float) -> str:
    if severity >= 0.85:
        return "critical"
    if severity >= 0.65:
        return "high"
    if severity >= 0.45:
        return "medium"
    return "low"


def _build_event_signature(event: Dict[str, Any]) -> str:
    payload = f"{event.get('id','')}|{event.get('severity',0)}|{event.get('confidence',0)}|{event.get('summary','')}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _dedupe_new_events(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    state = _load_json(EVENT_STATE_PATH, {"signatures_by_id": {}, "last_run_utc": None})
    old_map = state.get("signatures_by_id", {})
    new_map = {}
    new_or_changed = []
    unchanged = []
    for e in events:
        sig = _build_event_signature(e)
        event_id = e.get("id", sig)
        new_map[event_id] = sig
        if old_map.get(event_id) != sig:
            new_or_changed.append(e)
        else:
            unchanged.append(e)

    _save_json(
        EVENT_STATE_PATH,
        {"signatures_by_id": new_map, "last_run_utc": datetime.now(timezone.utc).isoformat()},
    )
    return {
        "new_or_changed_events": new_or_changed,
        "unchanged_events": unchanged,
        "event_driven_skip": len(new_or_changed) == 0,
        "state_counts": {"new_or_changed": len(new_or_changed), "unchanged": len(unchanged)},
    }


def _cache_lookup(cache_key: str) -> Optional[Dict[str, Any]]:
    cache = _load_json(PIPELINE_CACHE_PATH, {"items": {}})
    item = (cache.get("items") or {}).get(cache_key)
    if not item:
        return None
    return item if isinstance(item, dict) else None


def _cache_write(cache_key: str, value: Dict[str, Any]) -> None:
    cache = _load_json(PIPELINE_CACHE_PATH, {"items": {}})
    items = cache.setdefault("items", {})
    items[cache_key] = value
    cache["updated_utc"] = datetime.now(timezone.utc).isoformat()
    _save_json(PIPELINE_CACHE_PATH, cache)


def _append_workflow_log(record: Dict[str, Any]) -> None:
    store = _load_json(WORKFLOW_LOG_PATH, {"records": []})
    store.setdefault("records", []).append(record)
    _save_json(WORKFLOW_LOG_PATH, store)


def _estimate_tokens(events: List[Dict[str, Any]]) -> Dict[str, int]:
    context_chars = 0
    for e in events:
        compact = e.get("compact_context") or {}
        context_chars += len(json.dumps(compact, ensure_ascii=False))
    approx_prompt_tokens = max(1, context_chars // 4)
    approx_output_tokens = max(60, len(events) * 90)
    return {
        "approx_prompt_tokens": int(approx_prompt_tokens),
        "approx_output_tokens": int(approx_output_tokens),
        "approx_total_tokens": int(approx_prompt_tokens + approx_output_tokens),
    }


def _process_pipeline(events: List[Dict[str, Any]], company: Dict[str, Any]) -> Dict[str, Any]:
    stage1_threshold = 0.45
    stage1_kept = []
    dropped_count = 0
    for e in events:
        e["relevance_score"] = round(_rule_relevance(e, company), 3)
        if e["relevance_score"] >= stage1_threshold:
            stage1_kept.append(e)
        else:
            dropped_count += 1

    classification_calls = 0
    batch_size = 10
    for i in range(0, len(stage1_kept), batch_size):
        batch = stage1_kept[i:i + batch_size]
        classification_calls += 1
        for e in batch:
            e.update(_classify_signal(e))
            e["severity_tier"] = _tier_from_severity(float(e.get("severity", 0.0)))
            e["compact_context"] = {
                "id": e.get("id"),
                "type": e.get("disruption_type"),
                "region": e.get("region"),
                "severity": e.get("severity"),
                "affected": e.get("affected", []),
                "summary_short": _normalize_summary(str(e.get("summary", ""))),
            }

    stage3_kept = []
    cache_hits = 0
    cache_misses = 0
    llm_reasoning_calls = 0
    for e in stage1_kept:
        qualifies = (
            float(e.get("severity", 0.0)) >= 0.65
            and float(e.get("company_relevance", 0.0)) >= 0.8
            and e.get("severity_tier") in {"high", "critical"}
        )
        if not qualifies:
            continue

        cache_key = f"{e.get('disruption_type','other')}|{e.get('region','unknown')}|{datetime.now(timezone.utc).date().isoformat()}"
        cached = _cache_lookup(cache_key)
        if cached:
            cache_hits += 1
            e["reasoning_packet"] = cached
        else:
            cache_misses += 1
            llm_reasoning_calls += 1
            packet = {
                "input": e["compact_context"],
                "output_hint": "requires_mitigation_planning",
                "created_utc": datetime.now(timezone.utc).isoformat(),
            }
            _cache_write(cache_key, packet)
            e["reasoning_packet"] = packet
        stage3_kept.append(e)

    dedupe = _dedupe_new_events(stage3_kept)
    final_events = dedupe["new_or_changed_events"] if stage3_kept else []
    token_est = _estimate_tokens(stage3_kept)
    return {
        "events_for_risk": final_events,
        "all_candidate_events": stage3_kept,
        "pipeline_stats": {
            "raw_signals": len(events),
            "dropped_stage1": dropped_count,
            "stage1_pass": len(stage1_kept),
            "stage2_batched_classification_calls": classification_calls,
            "stage3_high_risk_pass": len(stage3_kept),
            "event_driven_new_or_changed": dedupe["state_counts"]["new_or_changed"],
            "event_driven_unchanged": dedupe["state_counts"]["unchanged"],
            "event_driven_skip": dedupe["event_driven_skip"],
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "estimated_llm_reasoning_calls": llm_reasoning_calls,
            "estimated_llm_calls_without_controls": len(events),
            **token_est,
        },
    }


def ingest_disruption_signals(company: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Perception layer with staged filtering and event-driven gating."""
    templates = _load_signal_templates()
    company = company or {}
    critical = set(company.get("critical_components", []))
    enriched: List[Dict[str, Any]] = []
    now = datetime.now(timezone.utc).isoformat()
    for idx, signal in enumerate(templates):
        s = dict(signal)
        s.setdefault("id", f"evt-{idx+1:03d}")
        s["ts"] = now
        s["company_relevance"] = 1.0 if critical.intersection(set(s.get("affected", []))) else 0.7
        enriched.append(s)

    pipeline = _process_pipeline(enriched, company)
    events_for_risk = pipeline["events_for_risk"]
    if events_for_risk:
        return events_for_risk
    # Keep compatibility: return at least one medium+ signal when nothing changed.
    fallback = sorted(enriched, key=lambda e: float(e.get("severity", 0.0)), reverse=True)[:1]
    for e in fallback:
        e["compact_context"] = {
            "id": e.get("id"),
            "type": e.get("type"),
            "region": e.get("region"),
            "severity": e.get("severity"),
            "affected": e.get("affected", []),
            "summary_short": _normalize_summary(str(e.get("summary", ""))),
        }
    return fallback


def derive_memory_feedback(company_name: str) -> Dict[str, Any]:
    """Extract simple feedback signals from historical disruption outcomes."""
    mem = _load_memory()
    related = [e for e in mem.get("events", []) if e.get("company_name") == company_name]
    if not related:
        return {"memory_events": 0, "risk_bias": 0.0, "suggested_focus": [], "avg_mitigation_success": None}

    critical_count = sum(1 for e in related if (e.get("risk", {}) or {}).get("risk_level") == "critical")
    escalation_count = sum(1 for e in related if e.get("approval_required"))
    success_scores = [float(e.get("mitigation_success_score")) for e in related if e.get("mitigation_success_score") is not None]
    avg_success = sum(success_scores) / len(success_scores) if success_scores else None

    risk_bias = min(0.08, (critical_count / max(1, len(related))) * 0.08)
    if avg_success is not None:
        # Outcome-based weighting: low mitigation success increases bias, strong success lowers it.
        if avg_success < 0.45:
            risk_bias += 0.03
        elif avg_success > 0.75:
            risk_bias = max(0.0, risk_bias - 0.02)

    focus = ["supplier_diversification"] if escalation_count > max(2, len(related) // 2) else []
    if avg_success is not None and avg_success < 0.55:
        focus.append("faster_escalation_playbooks")
    return {
        "memory_events": len(related),
        "risk_bias": round(min(0.12, max(0.0, risk_bias)), 4),
        "suggested_focus": list(dict.fromkeys(focus)),
        "avg_mitigation_success": None if avg_success is None else round(avg_success, 3),
    }


def score_risk(
    company: Dict[str, Any],
    events: List[Dict[str, Any]],
    memory_feedback: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Risk Intelligence Engine: procurement + logistics + inventory exposure score."""
    reasons = []
    procurement_risk = 0.0
    logistics_risk = 0.0
    inventory_risk = 0.0

    for e in events:
        event_risk = e["severity"] * e["confidence"] * e.get("company_relevance", 1.0)
        if e.get("category") == "procurement":
            procurement_risk = max(procurement_risk, event_risk)
        elif e.get("category") == "logistics":
            logistics_risk = max(logistics_risk, event_risk)
        else:
            inventory_risk = max(inventory_risk, event_risk)

        if "semiconductors" in e.get("affected", []) and "semiconductors" in company["critical_components"]:
            share = company["supplier_concentration"]["semiconductors"]["top_supplier_share"]
            reasons.append(f"Semiconductors impacted; top supplier share is {share:.0%} (high concentration).")
            procurement_risk += 0.08

    if company["lead_time_sensitivity"] == "high":
        reasons.append("High lead-time sensitivity increases disruption impact.")
        logistics_risk += 0.06

    buffer_days = company["inventory_policy"]["semiconductors_days_buffer"]
    if buffer_days >= 14:
        reasons.append("Inventory buffer >= 14 days reduces short-term risk.")
        inventory_risk -= 0.07
    else:
        reasons.append(f"Inventory buffer is only {buffer_days} days (thin buffer).")
        inventory_risk += 0.07

    risk = (
        max(0.0, min(1.0, procurement_risk)) * 0.40
        + max(0.0, min(1.0, logistics_risk)) * 0.35
        + max(0.0, min(1.0, inventory_risk)) * 0.25
    )
    if memory_feedback:
        risk += float(memory_feedback.get("risk_bias", 0.0))
        if memory_feedback.get("risk_bias", 0.0) > 0:
            reasons.append("Historical disruptions indicate persistent fragility; upward risk bias applied.")

    risk = max(0.0, min(1.0, risk))
    level = "low" if risk < 0.5 else "high" if risk < 0.8 else "critical"

    unique_reasons = list(dict.fromkeys(reasons))
    return {
        "risk_score": risk,
        "risk_level": level,
        "components": {
            "procurement_risk": round(max(0.0, min(1.0, procurement_risk)), 3),
            "logistics_risk": round(max(0.0, min(1.0, logistics_risk)), 3),
            "inventory_risk": round(max(0.0, min(1.0, inventory_risk)), 3),
        },
        "reasons": unique_reasons,
    }


def simulate_tradeoffs(company: Dict[str, Any], risk: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Planning engine: cost/service/resilience simulation."""
    critical_components = [str(c).lower() for c in company.get("critical_components", [])]
    semiconductor_dependent = "semiconductors" in critical_components
    if semiconductor_dependent:
        options = [
            {"action": "Pre-build 3 weeks of semiconductor stock", "cost_usd": 180000, "service_gain": 0.20, "resilience_gain": 0.25},
            {"action": "Reroute ocean freight via Cape of Good Hope", "cost_usd": 95000, "service_gain": 0.10, "resilience_gain": 0.15},
            {"action": "Qualify secondary semiconductor supplier (Vietnam)", "cost_usd": 220000, "service_gain": 0.12, "resilience_gain": 0.35},
            {"action": "Expedite critical lanes (air for 20%)", "cost_usd": 260000, "service_gain": 0.30, "resilience_gain": 0.10},
            {"action": "Renegotiate SLA flexibility + risk-sharing clauses", "cost_usd": 25000, "service_gain": 0.08, "resilience_gain": 0.12},
        ]
    else:
        options = [
            {"action": "Increase dual-source allocation for controllers across NA suppliers", "cost_usd": 85000, "service_gain": 0.14, "resilience_gain": 0.30},
            {"action": "Adjust safety stock for bearings and controllers to 21 days", "cost_usd": 120000, "service_gain": 0.18, "resilience_gain": 0.22},
            {"action": "Shift inbound mix from ocean to rail for at-risk lanes", "cost_usd": 70000, "service_gain": 0.11, "resilience_gain": 0.16},
            {"action": "Renegotiate supplier performance clauses and lead-time penalties", "cost_usd": 22000, "service_gain": 0.09, "resilience_gain": 0.13},
            {"action": "Activate regional substitute component policy for non-critical SKUs", "cost_usd": 45000, "service_gain": 0.10, "resilience_gain": 0.20},
        ]

    appetite = company.get("risk_appetite", "medium")
    cost_weight = 0.20 if appetite == "low" else 0.12 if appetite == "high" else 0.16
    resilience_weight = 0.65 if risk.get("risk_level") == "critical" else 0.55
    service_weight = 1.0 - resilience_weight

    for o in options:
        o["score"] = (
            (o["service_gain"] * service_weight)
            + (o["resilience_gain"] * resilience_weight)
            - (o["cost_usd"] / 1_000_000) * cost_weight
        )
        o["simulated_tradeoff"] = {
            "service_protection_pct": round(o["service_gain"] * 100, 1),
            "resilience_uplift_pct": round(o["resilience_gain"] * 100, 1),
            "cost_impact_usd": o["cost_usd"],
        }

    options.sort(key=lambda x: x["score"], reverse=True)
    return options


def generate_actions(company: Dict[str, Any], risk: Dict[str, Any], plan: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Autonomous action layer: drafts actions and execution triggers."""
    if not plan:
        plan = [{"action": "Monitor only", "cost_usd": 0, "service_gain": 0.0, "resilience_gain": 0.0, "score": 0.0}]
    top = dict(plan[0])
    escalation = risk["risk_level"] in {"high", "critical"}
    auto_mitigation = risk["risk_score"] >= 0.78
    primary_component = (company.get("critical_components") or ["critical_component"])[0]
    backup = dict(plan[1]) if len(plan) > 1 else None

    # Responsible-AI guardrails: constrain budget-intensive recommendations unless risk is critical.
    max_noncritical_cost = 200000
    guardrail_flags: List[str] = []
    if risk["risk_level"] != "critical" and float(top.get("cost_usd", 0)) > max_noncritical_cost and backup:
        guardrail_flags.append("budget_guardrail_applied")
        top = backup

    supplier_email = f"""Subject: Supply Resilience Negotiation Request — {company['company_name']}

Hi Supplier Team,

We are proactively mitigating disruption risk impacting component lead times.
To protect our production schedule, we’d like to discuss:
1) alternative shipping lanes / lead-time commitments
2) allocation guarantees for the next 6–8 weeks
3) pricing for expedited partial shipments if needed

Can you share your latest ETA outlook and risk mitigation options?

Thanks,
Operations — {company['company_name']}
"""

    executive_alert = f"""EXEC ALERT: {company['company_name']} Risk {risk['risk_level'].upper()} (score {risk['risk_score']:.2f})
Key reasons:
- {chr(10).join(risk['reasons'][:3])}

Recommended #1: {top['action']}
Estimated cost: ${top['cost_usd']:,}
"""

    tier_action = {
        "low": "store_signal_only",
        "medium": "dashboard_alert",
        "high": "ai_mitigation_planning",
        "critical": "executive_escalation",
    }.get(risk["risk_level"], "dashboard_alert")

    return {
        "recommended_top_action": top,
        "recommended_backup_action": backup,
        "draft_supplier_email": supplier_email,
        "draft_executive_alert": executive_alert if escalation else None,
        "erp_reorder_adjustment_flags": [
            {
                "component": primary_component,
                "recommended_reorder_increase_pct": min(30, 22 if risk["risk_level"] == "critical" else 10),
                "reason": "Lead-time disruption risk and thin buffer coverage.",
            }
        ],
        "preemptive_stock_build_recommendation": {
            "component": primary_component,
            "target_buffer_days": 21 if risk["risk_level"] == "critical" else 14,
            "status": "recommended",
        },
        "triggered_workflows": [
            {"workflow": "supplier_negotiation", "status": "triggered" if auto_mitigation else "drafted"},
            {"workflow": "erp_reorder_review", "status": "triggered" if auto_mitigation else "drafted"},
        ],
        "tiered_alert_action": tier_action,
        "guardrail_flags": guardrail_flags,
        "human_approval_required": escalation,
        "auto_execution_candidate": auto_mitigation,
    }


def run_cost_optimized_pipeline(company: Dict[str, Any]) -> Dict[str, Any]:
    raw_events = _load_signal_templates()
    now = datetime.now(timezone.utc).isoformat()
    critical = set(company.get("critical_components", []))
    enriched = []
    for idx, signal in enumerate(raw_events):
        s = dict(signal)
        s.setdefault("id", f"evt-{idx+1:03d}")
        s["ts"] = now
        s["company_relevance"] = 1.0 if critical.intersection(set(s.get("affected", []))) else 0.7
        enriched.append(s)

    pipeline = _process_pipeline(enriched, company)
    signal_profile = os.getenv("APP_SIGNAL_PROFILE", "default").strip().lower()
    if pipeline["events_for_risk"]:
        events_for_risk = pipeline["events_for_risk"]
    elif signal_profile == "critical" and pipeline.get("all_candidate_events"):
        # For demo critical scenarios, keep full high-risk context instead of single fallback event.
        events_for_risk = list(pipeline["all_candidate_events"])
    else:
        events_for_risk = sorted(enriched, key=lambda e: float(e.get("severity", 0.0)), reverse=True)[:1]
    for e in events_for_risk:
        if "compact_context" not in e:
            e["compact_context"] = {
                "id": e.get("id"),
                "type": e.get("type"),
                "region": e.get("region"),
                "severity": e.get("severity"),
                "affected": e.get("affected", []),
                "summary_short": _normalize_summary(str(e.get("summary", ""))),
            }
    return {
        "events_for_risk": events_for_risk,
        "pipeline_stats": pipeline["pipeline_stats"],
    }


def build_cost_value_report(
    risk: Dict[str, Any],
    pipeline_stats: Dict[str, Any],
    company: Dict[str, Any],
) -> Dict[str, Any]:
    estimated_daily_calls_without_controls = max(1, int(pipeline_stats.get("estimated_llm_calls_without_controls", 1)))
    estimated_daily_calls_with_controls = max(1, int(pipeline_stats.get("estimated_llm_reasoning_calls", 0)))
    call_reduction_pct = round(
        (1 - (estimated_daily_calls_with_controls / estimated_daily_calls_without_controls)) * 100, 1
    )
    unit_call_cost_usd = 0.04
    annual_ai_cost_usd = round(estimated_daily_calls_with_controls * unit_call_cost_usd * 365, 2)
    estimated_daily_tokens = int(pipeline_stats.get("approx_total_tokens", 0) * estimated_daily_calls_with_controls)
    token_cost_per_1k = 0.0015
    estimated_daily_token_cost_usd = round((estimated_daily_tokens / 1000.0) * token_cost_per_1k, 4)
    estimated_annual_token_cost_usd = round(estimated_daily_token_cost_usd * 365, 2)

    daily_revenue = 350000.0
    delay_days = 5 if risk.get("risk_level") in {"high", "critical"} else 2
    dependency = 0.65 if "semiconductors" in company.get("critical_components", []) else 0.45
    revenue_at_risk = round(daily_revenue * delay_days * dependency, 2)
    assumed_prevention_pct = 0.35 if risk.get("risk_level") in {"high", "critical"} else 0.15
    value_saved = round(revenue_at_risk * assumed_prevention_pct, 2)
    roi = round(value_saved / max(1.0, annual_ai_cost_usd), 2)

    return {
        "estimated_daily_calls_without_controls": estimated_daily_calls_without_controls,
        "estimated_daily_calls_with_controls": estimated_daily_calls_with_controls,
        "estimated_call_reduction_pct": call_reduction_pct,
        "estimated_annual_ai_cost_usd": annual_ai_cost_usd,
        "estimated_daily_tokens": estimated_daily_tokens,
        "estimated_daily_token_cost_usd": estimated_daily_token_cost_usd,
        "estimated_annual_token_cost_usd": estimated_annual_token_cost_usd,
        "estimated_revenue_at_risk_usd": revenue_at_risk,
        "estimated_revenue_saved_usd": value_saved,
        "estimated_roi_multiple": roi,
        "formula_notes": {
            "revenue_at_risk": "daily_revenue * delay_days * component_dependency",
            "controls": "rule filters + batching + cache + high-risk reasoning gate",
        },
    }


def estimate_mitigation_success_score(
    risk: Dict[str, Any],
    actions: Dict[str, Any],
    cost_value_report: Dict[str, Any],
) -> float:
    """Heuristic post-cycle success score for memory-weighted learning."""
    base = 0.55
    if risk.get("risk_level") == "critical":
        base -= 0.15
    if actions.get("tiered_alert_action") in {"ai_mitigation_planning", "executive_escalation"}:
        base += 0.08
    if actions.get("guardrail_flags"):
        base += 0.05
    if actions.get("auto_execution_candidate"):
        base += 0.04

    roi = float(cost_value_report.get("estimated_roi_multiple", 0.0))
    if roi > 10:
        base += 0.07
    elif roi < 2:
        base -= 0.08
    return round(max(0.0, min(1.0, base)), 3)


def log_mock_workflow_execution(
    company: Dict[str, Any],
    risk: Dict[str, Any],
    actions: Dict[str, Any],
) -> Dict[str, Any]:
    """Emit deterministic mock integration logs to prove workflow execution paths."""
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "company_id": company.get("company_id"),
        "company_name": company.get("company_name"),
        "risk_level": risk.get("risk_level"),
        "tiered_alert_action": actions.get("tiered_alert_action"),
        "workflows": actions.get("triggered_workflows", []),
        "mock_provider_results": [
            {"provider": "webhook", "status": "success", "code": 200},
            {"provider": "slack", "status": "success", "code": 200},
        ],
    }
    _append_workflow_log(record)
    return {"logged": True, "providers": 2, "path": WORKFLOW_LOG_PATH}


def analyze_custom_profile(profile: Any) -> Dict[str, Any]:
    """
    Analyze an arbitrary company profile provided at runtime (dict or JSON string).
    This avoids fallback to predefined mock profiles when users paste a custom company.
    """
    if isinstance(profile, str):
        try:
            profile = json.loads(profile)
        except json.JSONDecodeError as exc:
            return {"error": f"Invalid JSON profile: {exc}"}
    if not isinstance(profile, dict):
        return {"error": "Profile must be a JSON object or dict."}

    default_profile = get_company_profile("de_semiconductor_auto")
    company = _deep_merge(default_profile, profile)
    company.setdefault("company_id", "custom_company")
    company.setdefault("company_name", "Custom Company")
    company.setdefault("critical_components", ["semiconductors"])
    company.setdefault("risk_appetite", "medium")

    memory_feedback = derive_memory_feedback(company.get("company_name", "Custom Company"))
    optimized_pipeline = run_cost_optimized_pipeline(company)
    events = optimized_pipeline["events_for_risk"]
    pipeline_stats = optimized_pipeline["pipeline_stats"]
    risk = score_risk(company, events, memory_feedback=memory_feedback)
    plan = simulate_tradeoffs(company, risk)
    actions = generate_actions(company, risk, plan)
    cost_value_report = build_cost_value_report(risk, pipeline_stats, company)
    workflow_execution_log = log_mock_workflow_execution(company, risk, actions)
    mitigation_success_score = estimate_mitigation_success_score(risk, actions, cost_value_report)

    memory_event = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "company_id": company.get("company_id"),
        "company_name": company.get("company_name"),
        "top_event": events[0] if events else None,
        "pipeline_stats": pipeline_stats,
        "risk": risk,
        "top_action": actions.get("recommended_top_action"),
        "cost_value_report": cost_value_report,
        "mitigation_success_score": mitigation_success_score,
        "workflow_execution_log": workflow_execution_log,
        "approval_required": actions.get("human_approval_required", False),
    }
    memory_write = write_memory(memory_event)

    return {
        "company": company,
        "events": events,
        "memory_feedback": memory_feedback,
        "pipeline_stats": pipeline_stats,
        "risk": risk,
        "plan": plan,
        "actions": actions,
        "cost_value_report": cost_value_report,
        "workflow_execution_log": workflow_execution_log,
        "memory_write": memory_write,
    }


def write_memory(event_summary: Dict[str, Any]) -> Dict[str, Any]:
    mem = _load_memory()
    mem["events"].append(event_summary)
    _save_memory(mem)
    return {"saved": True, "events_count": len(mem["events"])}


def read_memory() -> Dict[str, Any]:
    return _load_memory()
