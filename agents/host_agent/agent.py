from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import json


import os 
from dotenv import load_dotenv
load_dotenv()


os.environ['GEMINI_API_KEY'] = os.getenv('GEMINI_API_KEY')
os.environ['REQUESTS_CA_BUNDLE'] = ''  # Empty string disables verification


#Step 1: Host agent  
host_agent = Agent(
    name="host_agent",
    model=LiteLlm("gemini/gemini-pro"),
    description="Coordinates travel planning by calling flight, stay, and activity agents.",
    instruction="You are the host agent responsible for orchestrating trip planning tasks. "
                "You call external agents to gather flights, stays, and activities, then return a final result."
)

#Step 2: Session management
session_service = InMemorySessionService()
runner = Runner(
    agent=host_agent,
    app_name="host_app",
    session_service=session_service
)
USER_ID = "user_host"
SESSION_ID = "session_host"



#Step 3: Executing the agent logic


async def execute(request):
    # Ensure session exists
    session_service.create_session(
        app_name="host_app",
        user_id=USER_ID,
        session_id=SESSION_ID
    )
    prompt = (
        f"Plan a trip to {request['destination']} from {request['start_date']} to {request['end_date']} "
        f"within a total budget of {request['budget']}. Call the flights, stays, and activities agents for results."
    )
    message = types.Content(role="user", parts=[types.Part(text=prompt)])
    async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=message):
        if event.is_final_response():
            return {"summary": event.content.parts[0].text}