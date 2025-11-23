import os

from google.genai import types
from google.adk.models.google_llm import Gemini
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

from .tools_library import create_ticket_and_notify_slack, get_ticket_details, search_knowledge_base

from dotenv import load_dotenv


try:
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
    print("✅ Gemini API key setup complete.")
except Exception as e:
    print(f"🔑 Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}")


retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

# Wrap the specific tools as FunctionTools for the agent to use
chained_tool_instance = FunctionTool(create_ticket_and_notify_slack)
get_ticket_tool_instance = FunctionTool(get_ticket_details)
rag_tool_instance = FunctionTool(search_knowledge_base)

# Ticket Agent
ticket_agent = LlmAgent(
    name="TicketAgent",
    description="An agent for managing support tickets and internal notifications.",
    instruction="""
    You are an expert Ticket Management Agent. 
    Use the 'create_ticket_and_notify_slack' tool immediately when the user requests to create a new ticket. 
    Ensure you capture the 'description' from the user's request to call the tool correctly. 
    Do not try to call a slack tool separately. The process is automatic within the provided tool.
    """,
    tools=[chained_tool_instance, get_ticket_tool_instance]
)

# Rag Agent
rag_agent = LlmAgent(
    name="RagAgent",
    description="An agent for retrieving relevant information to answer questions and guide users through the application clearly and accurately.",
    instruction="Extract all necessary information from the user query using search_knowledge_base and summarize.",
    tools=[rag_tool_instance]
)

# Root Agent
root_agent = LlmAgent(
    name="CoordinatorAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
        Coordinate tasks efficiently. 
        If the user asks about general knowledge, use the RagAgent. 
        Always first try to answer from RagAgent. 
        If they mention creating, viewing, or managing 'tickets', delegate immediately to the TicketAgent.
    """,
    sub_agents=[rag_agent, ticket_agent]
)