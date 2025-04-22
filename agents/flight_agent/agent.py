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

#Step 1: Flight agent  
flight_agent = Agent(
    name="flight_agent",
    model=LiteLlm("gemini/gemini-2.0-flash"),
    description="Suggests flight options for users traveling to a destination.",
    instruction=(
        "Given an origin, destination, and dates, suggest 2-3 flight options. "
        "For each flight option, provide airline name, departure/arrival times, price estimate, and duration. "
        "Respond in plain English. Keep it concise and well-formatted."
    )
)

#Step 2: Session management

session_service = InMemorySessionService()
runner = Runner(
    agent=flight_agent,
    app_name="flight_app",
    session_service=session_service
)
USER_ID = "user_flights"
SESSION_ID = "session_flights"

#Step 3: Executing the agent logic

async def execute(request):
    session_service.create_session(
        app_name="flight_app",
        user_id=USER_ID,
        session_id=SESSION_ID
    )
    prompt = (
        f"User is flying from {request['origin']} to {request['destination']} "
        f"departing on {request['start_date']} and returning on {request['end_date']}, "
        f"with a budget of {request['budget']}. Suggest 2-3 flight options, each with airline name, "
        f"departure/arrival times, price estimate, and duration. "
        f"Respond in JSON format using the key 'flights' with a list of flight objects."
    )
    message = types.Content(role="user", parts=[types.Part(text=prompt)])
    async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=message):
        if event.is_final_response():
            response_text = event.content.parts[0].text
            try:
                parsed = json.loads(response_text)
                if "flights" in parsed and isinstance(parsed["flights"], list):
                    return {"flights": parsed["flights"]}
                else:
                    print("'flights' key missing or not a list in response JSON")
                    return {"flights": response_text}  # fallback to raw text
            except json.JSONDecodeError as e:
                print("JSON parsing failed:", e)
                print("Response content:", response_text)
                return {"flights": response_text}  # fallback to raw text