import asyncio
import logging
from typing import List, Dict, Any
from datetime import datetime, timezone

import google.generativeai as genai
from agent.config import settings
from agent.models.supply_chain import Company

logger = logging.getLogger(__name__)

class ERPMonitoringService:
    def __init__(self):
        genai.configure(api_key=settings.google_api_key)
        # Using Flash for speed in monitoring tasks
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    async def monitor_inventory_levels(self, company: Company) -> Dict[str, Any]:
        """Monitor inventory levels and identify potential issues."""
        # Mock ERP data (Replace with real DB/API calls later)
        mock_inv = {
            'total_items': 1250, 'low_stock_items': 45, 'out_of_stock_items': 8,
            'excess_stock_items': 23, 'days_of_supply': 42
        }
        
        # Scoring Logic (0.0 to 1.0)
        health = max(0, 1 - (
            (mock_inv['out_of_stock_items'] / mock_inv['total_items']) * 0.4 +
            (mock_inv['low_stock_items'] / mock_inv['total_items']) * 0.3
        ))
        
        issues = []
        if mock_inv['out_of_stock_items'] > 0:
            issues.append(f"{mock_inv['out_of_stock_items']} items out of stock")
            
        return {
            'inventory_health_score': round(health, 2),
            'metrics': mock_inv,
            'identified_issues': issues,
            'last_updated': datetime.now(timezone.utc).isoformat()
        }

    async def analyze_erp_signals(self, company: Company) -> Dict[str, Any]:
        """Runs all ERP checks in parallel and uses AI for recommendations."""
        
        # FIXED: Run all monitors at once for high performance
        results = await asyncio.gather(
            self.monitor_inventory_levels(company),
            self.monitor_purchase_orders(company),
            self.monitor_demand_signals(company)
        )
        
        inventory, pos, demand = results
        
        # Calculate Overall Health
        overall_health = (
            inventory['inventory_health_score'] * 0.4 +
            pos['po_health_score'] * 0.3 +
            demand['demand_stability_score'] * 0.3
        )
        
        # FIXED: Use Gemini for an 'Intelligent' recommendation
        ai_advice = await self._generate_ai_summary(company, results)
        
        return {
            'company_id': company.id,
            'overall_erp_health': round(overall_health, 2),
            'inventory': inventory,
            'purchase_orders': pos,
            'demand': demand,
            'ai_recommendations': ai_advice,
            'status': "CRITICAL" if overall_health < 0.6 else "STABLE"
        }

    async def _generate_ai_summary(self, company: Company, results: List[Dict]) -> List[str]:
        """Uses Gemini to synthesize recommendations across all ERP signals."""
        prompt = f"Company {company.name} has these ERP signals: {results}. Give 3 concise mitigation steps."
        try:
            response = await self.model.generate_content_async(prompt)
            return [line.strip("- ") for line in response.text.split('\n') if line.strip()][:3]
        except Exception:
            return ["Review safety stock", "Contact late suppliers"]

    async def monitor_purchase_orders(self, company: Company) -> Dict[str, Any]:
        mock_pos = {
            "open_po_count": 82,
            "late_po_count": 9,
            "on_time_rate": 0.89,
        }
        po_health = max(0.0, min(1.0, mock_pos["on_time_rate"] - (mock_pos["late_po_count"] / 100)))
        return {
            "po_health_score": round(po_health, 2),
            "metrics": mock_pos,
            "identified_issues": ["Late PO concentration on critical SKUs"] if mock_pos["late_po_count"] > 5 else [],
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

    async def monitor_demand_signals(self, company: Company) -> Dict[str, Any]:
        mock_demand = {
            "forecast_variance_pct": 0.14,
            "demand_spike_items": 6,
        }
        stability = max(0.0, min(1.0, 1.0 - mock_demand["forecast_variance_pct"]))
        return {
            "demand_stability_score": round(stability, 2),
            "metrics": mock_demand,
            "identified_issues": ["Demand volatility on top-selling items"] if mock_demand["forecast_variance_pct"] > 0.1 else [],
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
