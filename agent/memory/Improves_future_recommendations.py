from __future__ import annotations

from typing import Any, Dict

from .reflection_engine import ReflectionEngine


async def improve_future_recommendations(company_id: str) -> Dict[str, Any]:
    engine = ReflectionEngine()
    return await engine.get_company_insights(company_id)
