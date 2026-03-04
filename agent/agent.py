import os

from dotenv import load_dotenv
from google.adk.agents import LlmAgent

from .tools import (
    analyze_custom_profile,
    get_company_profile,
    ingest_disruption_signals,
    score_risk,
    simulate_tradeoffs,
    generate_actions,
    write_memory,
    read_memory,
)

load_dotenv()

root_agent = LlmAgent(
    name="autonomous_supply_chain_resilience_agent",
    model=os.getenv("GOOGLE_LLM_MODEL", "gemini-2.5-flash"),
    instruction=(
        "You are an Autonomous Supply Chain Resilience Agent for mid-market manufacturers.\n"
        "Intent handling:\n"
        "- If user sends a greeting/small talk (e.g., 'hi', 'hello', 'thanks'), reply briefly and DO NOT call tools.\n"
        "- Run the full supply-chain pipeline only when the user asks for analysis, risk scoring, planning, actions, or a cycle run.\n\n"
        "When pipeline is requested, follow this sequence:\n"
        "1) Perceive: ingest disruption signals.\n"
        "2) Personalize: load the company profile.\n"
        "3) Reason: compute risk score and explain the reasons.\n"
        "4) Plan: simulate trade-offs and rank mitigation options.\n"
        "5) Act: draft operational actions (emails/alerts/reorder suggestions).\n"
        "6) Memory: save the event, decision, and rationale.\n\n"
        "If user provides a custom company profile JSON, use analyze_custom_profile.\n\n"
        "Rules:\n"
        "- Be proactive: if risk >= 0.8, recommend escalation.\n"
        "- Be explainable: always provide a short rationale.\n"
        "- Human-in-the-loop: mark actions requiring approval.\n"
        "- Avoid unnecessary tool calls to minimize API cost.\n"
    ),
    tools=[
        analyze_custom_profile,
        get_company_profile,
        ingest_disruption_signals,
        score_risk,
        simulate_tradeoffs,
        generate_actions,
        write_memory,
        read_memory,
    ],
)
