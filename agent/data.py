import feedparser
from datetime import datetime

# ------------------------
# MOCK INTERNAL DATA
# ------------------------

def get_company_profile():
    return {
        "company_name": "MidMarket AutoParts Co.",
        "daily_revenue_usd": 250000,
        "critical_suppliers": {
            "China": "Shenzhen Semi Ltd",
            "Japan": "Kanto Bearings"
        }
    }


# ------------------------
# LIVE RSS DISRUPTION FEED
# ------------------------

def fetch_live_disruptions(query="supply chain disruption", limit=5):
    url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}"
    feed = feedparser.parse(url)

    results = []
    for entry in feed.entries[:limit]:
        results.append({
            "headline": entry.title,
            "timestamp": entry.get("published", str(datetime.utcnow())),
            "region": infer_region(entry.title)
        })

    return results


def infer_region(text):
    text = text.lower()
    if "china" in text:
        return "China"
    if "japan" in text:
        return "Japan"
    if "singapore" in text:
        return "Singapore"
    if "mexico" in text:
        return "Mexico"
    return "Global"


# ------------------------
# RISK SCORING LOGIC
# ------------------------

def assess_risk(disruption):
    profile = get_company_profile()

    base_probability = 0.3
    region = disruption["region"]

    if region in profile["critical_suppliers"]:
        base_probability += 0.3

    estimated_downtime = 5 if region != "Global" else 2

    revenue_at_risk = (
        profile["daily_revenue_usd"] *
        estimated_downtime *
        base_probability
    )

    return {
        "region": region,
        "probability": round(base_probability, 2),
        "estimated_downtime_days": estimated_downtime,
        "revenue_at_risk_usd": int(revenue_at_risk)
    }


# ------------------------
# MITIGATION SIMULATION
# ------------------------

def simulate_tradeoffs(risk):
    options = [
        {"action": "Build safety stock", "cost_usd": 80000},
        {"action": "Expedite shipping", "cost_usd": 45000},
        {"action": "Do nothing", "cost_usd": 0}
    ]

    for option in options:
        option["score"] = (
            risk["revenue_at_risk_usd"] / 100000
            - option["cost_usd"] / 50000
        )

    return sorted(options, key=lambda x: x["score"], reverse=True)