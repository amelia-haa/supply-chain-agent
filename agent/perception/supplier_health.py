import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

import google.generativeai as genai
from agent.config import settings
from agent.models.supply_chain import Supplier

# Initialize Logging
logger = logging.getLogger(__name__)

class SupplierMonitoringService:
    def __init__(self):
        """Initialize with Gemini 1.5 Flash for high-speed risk assessment."""
        if not settings.google_api_key:
            logger.error("❌ GOOGLE_API_KEY missing in .env file")
            
        genai.configure(api_key=settings.google_api_key)
        # Flash is better for high-frequency monitoring tasks
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def _clamp(self, value: float) -> float:
        """Ensures scores stay between 0.0 and 1.0."""
        return max(0.0, min(1.0, value))

    async def assess_supplier_risk(self, supplier: Supplier) -> Dict[str, Any]:
        """
        Performs a deep-dive risk assessment.
        Combines deterministic scoring with AI-driven insights.
        """
        try:
            # 1. Deterministic Risk Scoring (The "Math" Brain)
            # Higher = More Risk
            performance_risk = self._clamp(1.0 - supplier.on_time_delivery_rate)
            quality_risk = self._clamp(1.0 - supplier.quality_score)
            financial_risk = self._clamp(1.0 - supplier.financial_health_score)
            
            # Weighted Composite Score
            risk_score = (
                performance_risk * 0.35 +
                quality_risk * 0.25 +
                financial_risk * 0.20 +
                supplier.geopolitical_risk_score * 0.15 +
                supplier.climate_risk_score * 0.05
            )
            risk_score = self._clamp(risk_score)

            # 2. Risk Categorization
            if risk_score > 0.75:
                category = "CRITICAL"
            elif risk_score > 0.45:
                category = "HIGH"
            elif risk_score > 0.25:
                category = "MEDIUM"
            else:
                category = "LOW"

            # 3. Identify Drivers (For Transparency)
            drivers = []
            if performance_risk > 0.2: drivers.append("Unstable delivery performance")
            if quality_risk > 0.15: drivers.append("Quality control variance")
            if financial_risk > 0.4: drivers.append("Financial vulnerability detected")

            # 4. AI-Powered Mitigation Strategy
            recommendations = await self._generate_ai_mitigation(supplier, category, drivers)

            return {
                "supplier_id": supplier.id,
                "supplier_name": supplier.name,
                "risk_score": round(risk_score, 3),
                "risk_category": category,
                "primary_risk_drivers": drivers or ["Stable operations"],
                "ai_recommendations": recommendations,
                "last_assessment_utc": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to assess supplier {supplier.id}: {str(e)}")
            return {"error": "Assessment failed", "supplier_id": supplier.id}

    async def _generate_ai_mitigation(self, supplier: Supplier, category: str, drivers: List[str]) -> List[str]:
        """Generates real-time mitigation steps using Gemini."""
        if category == "LOW":
            return ["Maintain standard quarterly reviews."]

        prompt = (
            f"Supplier {supplier.name} has a {category} risk rating. "
            f"Key issues: {', '.join(drivers)}. "
            f"Provide 2 short, professional mitigation actions for a supply chain manager."
        )

        try:
            response = await self.model.generate_content_async(prompt)
            # Clean up output into a list of strings
            lines = [line.strip("- ").strip() for line in response.text.split('\n') if line.strip()]
            return lines[:2] # Keep it concise for the UI
        except Exception:
            # Safe Fallback if API fails
            return ["Initiate secondary supplier audit.", "Review safety stock buffers."]

    async def monitor_batch(self, suppliers: List[Supplier]) -> List[Dict[str, Any]]:
        """Processes multiple suppliers in parallel for maximum efficiency."""
        tasks = [self.assess_supplier_risk(s) for s in suppliers]
        return await asyncio.gather(*tasks)