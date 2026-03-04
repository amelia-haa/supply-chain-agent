from __future__ import annotations

import json
import os
import re
import ssl
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple


RSS_SOURCES = [
    "https://feeds.reuters.com/reuters/worldNews",
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "https://www.aljazeera.com/xml/rss/all.xml",
]
JSON_SOURCES = [
    # Public, no-key weather alerts (US-focused).
    "https://api.weather.gov/alerts/active",
]

LIVE_OUTPUT_PATH = "data/live_disruption_signals.json"


def _safe_fetch(url: str, timeout: int = 10, retries: int = 2, allow_insecure_ssl: bool = False) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "supply-chain-agent/1.0"})
    attempt = 0
    while True:
        try:
            if allow_insecure_ssl:
                ctx = ssl._create_unverified_context()
                with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
                    return resp.read().decode("utf-8", errors="ignore")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", errors="ignore")
        except Exception:
            attempt += 1
            if attempt > retries:
                raise
            time.sleep(min(1.5, 0.4 * attempt))


def _safe_fetch_json(
    url: str,
    timeout: int = 10,
    retries: int = 2,
    allow_insecure_ssl: bool = False,
) -> Dict[str, Any]:
    raw = _safe_fetch(url, timeout=timeout, retries=retries, allow_insecure_ssl=allow_insecure_ssl)
    return json.loads(raw)


def _extract_items(rss_xml: str) -> List[Dict[str, str]]:
    root = ET.fromstring(rss_xml)
    items: List[Dict[str, str]] = []
    for node in root.findall(".//item"):
        title = (node.findtext("title") or "").strip()
        desc = (node.findtext("description") or "").strip()
        link = (node.findtext("link") or "").strip()
        pub_date = (node.findtext("pubDate") or "").strip()
        if title:
            items.append({"title": title, "description": desc, "link": link, "pub_date": pub_date})
    return items


def _classify_to_signal(item: Dict[str, str], idx: int) -> Dict[str, Any] | None:
    text = f"{item.get('title', '')} {item.get('description', '')}".lower()
    region = "Global"
    if "red sea" in text:
        region = "Red Sea"
    elif "taiwan" in text or "china" in text or "east asia" in text:
        region = "East Asia"
    elif "europe" in text:
        region = "Europe"
    elif "mexico" in text or "north america" in text:
        region = "North America"

    if any(k in text for k in ["shipping", "port", "freight", "container", "suez"]):
        event_type = "shipping_disruption"
        category = "logistics"
        severity = 0.72
    elif any(k in text for k in ["semiconductor", "chip", "electronics supply"]):
        event_type = "semiconductor_shortage"
        category = "procurement"
        severity = 0.78
    elif any(k in text for k in ["flood", "storm", "hurricane", "wildfire", "earthquake"]):
        event_type = "climate_event"
        category = "logistics"
        severity = 0.68
    elif any(k in text for k in ["sanction", "export control", "tariff", "geopolitical", "conflict", "war"]):
        event_type = "geopolitical_issue"
        category = "procurement"
        severity = 0.7
    elif any(k in text for k in ["bankruptcy", "insolvency", "default", "debt restructuring"]):
        event_type = "supplier_financial_alert"
        category = "procurement"
        severity = 0.66
    else:
        return None

    summary = re.sub(r"\s+", " ", item.get("title", "")).strip()
    return {
        "id": f"live-{idx:04d}",
        "type": event_type,
        "region": region,
        "severity": severity,
        "confidence": 0.62,
        "summary": summary[:200],
        "affected": ["ocean_freight", "semiconductors"] if category == "logistics" else ["semiconductors", "controllers"],
        "category": category,
        "source": item.get("link"),
        "source_ts": item.get("pub_date"),
        "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _map_noaa_alerts(payload: Dict[str, Any], idx_start: int = 1) -> Tuple[List[Dict[str, Any]], int]:
    out: List[Dict[str, Any]] = []
    idx = idx_start
    for feature in payload.get("features", []):
        props = feature.get("properties", {}) or {}
        headline = str(props.get("headline") or props.get("event") or "Weather alert").strip()
        if not headline:
            continue
        severity_raw = str(props.get("severity") or "").lower()
        urgency_raw = str(props.get("urgency") or "").lower()
        area = str(props.get("areaDesc") or "US").strip()
        desc = str(props.get("description") or "")
        summary = re.sub(r"\s+", " ", headline)

        sev = 0.6
        if "extreme" in severity_raw:
            sev = 0.9
        elif "severe" in severity_raw:
            sev = 0.78
        elif "moderate" in severity_raw:
            sev = 0.64
        if "immediate" in urgency_raw:
            sev = min(0.95, sev + 0.08)
        elif "expected" in urgency_raw:
            sev = min(0.9, sev + 0.03)

        out.append(
            {
                "id": f"live-noaa-{idx:04d}",
                "type": "climate_event",
                "region": area[:120] if area else "US",
                "severity": round(sev, 2),
                "confidence": 0.72,
                "summary": summary[:200],
                "affected": ["ocean_freight", "finished_goods"],
                "category": "logistics",
                "source": str(props.get("web") or "https://api.weather.gov/alerts/active"),
                "source_ts": str(props.get("sent") or ""),
                "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_type": "noaa_alert",
                "raw_event": str(props.get("event") or ""),
                "description_short": re.sub(r"\s+", " ", desc)[:280],
            }
        )
        idx += 1
    return out, idx


def fetch_live_disruption_signals(max_items: int = 30) -> Dict[str, Any]:
    raw_items: List[Dict[str, str]] = []
    errors: List[str] = []
    allow_insecure_ssl = str(os.environ.get("ALLOW_INSECURE_SSL_FETCH", "false")).strip().lower() in {
        "1",
        "true",
        "yes",
    }
    include_noaa = str(os.environ.get("LIVE_INCLUDE_NOAA", "true")).strip().lower() in {"1", "true", "yes"}
    source_counts: Dict[str, int] = {"rss_items": 0, "noaa_alerts": 0}

    for url in RSS_SOURCES:
        try:
            xml = _safe_fetch(url, allow_insecure_ssl=allow_insecure_ssl)
            parsed = _extract_items(xml)
            raw_items.extend(parsed)
            source_counts["rss_items"] += len(parsed)
        except Exception as exc:
            errors.append(f"{url}: {exc}")

    if include_noaa:
        for url in JSON_SOURCES:
            try:
                payload = _safe_fetch_json(url, allow_insecure_ssl=allow_insecure_ssl)
                mapped, _ = _map_noaa_alerts(payload, idx_start=1)
                source_counts["noaa_alerts"] += len(mapped)
                # Reuse classification/dedupe pipeline by converting to item-like inputs.
                for m in mapped:
                    raw_items.append(
                        {
                            "title": m.get("summary", ""),
                            "description": m.get("description_short", ""),
                            "link": m.get("source", ""),
                            "pub_date": m.get("source_ts", ""),
                            "_pre_mapped_signal": m,
                        }
                    )
            except Exception as exc:
                errors.append(f"{url}: {exc}")

    signals: List[Dict[str, Any]] = []
    seen = set()
    for idx, item in enumerate(raw_items, start=1):
        pre_mapped = item.get("_pre_mapped_signal") if isinstance(item, dict) else None
        signal = pre_mapped if isinstance(pre_mapped, dict) else _classify_to_signal(item, idx)
        if not signal:
            continue
        key = (signal["type"], signal["region"], signal["summary"][:80])
        if key in seen:
            continue
        seen.add(key)
        signals.append(signal)
        if len(signals) >= max_items:
            break

    return {
        "meta": {
            "source_mode": "live_rss",
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
            "raw_items": len(raw_items),
            "signals": len(signals),
            "allow_insecure_ssl_fetch": allow_insecure_ssl,
            "include_noaa": include_noaa,
            "source_counts": source_counts,
            "errors": errors,
        },
        "signals": signals,
    }


def write_live_signals(output_path: str = LIVE_OUTPUT_PATH, max_items: int = 30) -> Dict[str, Any]:
    payload = fetch_live_disruption_signals(max_items=max_items)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload
