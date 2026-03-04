from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from agent.orchestrator import AutonomousSupplyChainOrchestrator


async def run_autonomous_loop(
    cycles: int = 1,
    interval_seconds: int = 30,
    company_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    orchestrator = AutonomousSupplyChainOrchestrator()
    outputs: List[Dict[str, Any]] = []
    company_ids = company_ids or ["de_semiconductor_auto"]

    for idx in range(cycles):
        for company_id in company_ids:
            result = orchestrator.run_cycle(company_id=company_id).to_dict()
            result["cycle_index"] = idx + 1
            outputs.append(result)

        if idx < cycles - 1:
            await asyncio.sleep(interval_seconds)

    return outputs
