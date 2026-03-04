from google.adk.agents import LlmAgent
from google.adk.tools import agent_tool
from google.adk.tools.google_search_tool import GoogleSearchTool
from google.adk.tools import url_context

supply_chain_agent__google_search_agent = LlmAgent(
  name='supply_chain_agent__google_search_agent',
  model='gemini-2.5-flash',
  description=(
      'Agent specialized in performing Google searches.'
  ),
  sub_agents=[],
  instruction='Use the GoogleSearchTool to find information on the web.',
  tools=[
    GoogleSearchTool()
  ],
)
supply_chain_agent__url_context_agent = LlmAgent(
  name='supply_chain_agent__url_context_agent',
  model='gemini-2.5-flash',
  description=(
      'Agent specialized in fetching content from URLs.'
  ),
  sub_agents=[],
  instruction='Use the UrlContextTool to retrieve content from provided URLs.',
  tools=[
    url_context
  ],
)
root_agent = LlmAgent(
  name='supply_chain_agent_',
  model='gemini-2.5-flash',
  description=(
      'Agent to help interact with my data.'
  ),
  sub_agents=[],
  instruction='',
  tools=[
    agent_tool.AgentTool(agent=supply_chain_agent__google_search_agent),
    agent_tool.AgentTool(agent=supply_chain_agent__url_context_agent)
  ],
)