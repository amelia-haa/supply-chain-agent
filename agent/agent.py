from google.adk.agents import LlmAgent
from dotenv import load_dotenv

load_dotenv()

assistant = LlmAgent(
    name="supply_chain_agent",
    model="gemini-2.5-flash",
    description="AI Supply Chain Resilience Co-Pilot",
    instruction="You are an AI operations co-pilot for mid-market manufacturers."
)
