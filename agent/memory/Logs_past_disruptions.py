from __future__ import annotations

from typing import Any, Dict

from .reflection_engine import ReflectionEngine


async def log_past_disruption(company_id: str, disruption: Dict[str, Any]) -> None:
    engine = ReflectionEngine()
    await engine.log_disruption(company_id, disruption)
