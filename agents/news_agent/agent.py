
import json
import os
import httpx
from typing import Any, Annotated, Dict
import traceback 

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session 
from google.adk.tools import FunctionTool
from google.genai import types

from dotenv import load_dotenv

load_dotenv()

# --- Environment Variable Setup ---
gemini_api_key = os.getenv('GEMINI_API_KEY')
news_api_key = os.getenv('NEWS_API_KEY')

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")
if not news_api_key:
    raise ValueError("NEWS_API_KEY environment variable not set.")

os.environ['GEMINI_API_KEY'] = gemini_api_key
os.environ['REQUESTS_CA_BUNDLE'] = ''

# --- Constants ---
NEWS_API_BASE = "https://newsapi.org/v2"
USER_AGENT = "adk-news-agent/1.0"
APP_NAME_CONST = "news_app"
USER_ID_CONST = "user_news"
SESSION_ID_CONST = "session_news" # This ID will be reused, but the session data is ephemeral

# --- Tool Function ---
async def search_news(
    query: Annotated[str, "The topic, keyword, or location to search news for (e.g., 'Paris tourism', 'London museums')."],
    start_date: Annotated[str, "The start date for the news search in<x_bin_880>-MM-DD format."],
    end_date: Annotated[str, "The end date for the news search in<x_bin_880>-MM-DD format."]
) -> Dict[str, Any]:


    print(f"Tool Function 'search_news' called with query='{query}', start_date='{start_date}', end_date='{end_date}'")
    endpoint = f"{NEWS_API_BASE}/everything"
    headers = {"User-Agent": USER_AGENT, "X-Api-Key": news_api_key}
    params = {
        "q": query,
        "from": start_date,
        "to": end_date,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 5,
    }

    try:
        async with httpx.AsyncClient(verify=False if os.environ.get('REQUESTS_CA_BUNDLE') == '' else True) as client:
            response = await client.get(endpoint, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "ok":
                articles = data.get("articles", [])
                simplified_articles = [
                    {"title": article.get("title"), "description": article.get("description"),
                     "url": article.get("url"), "publishedAt": article.get("publishedAt")}
                    for article in articles]
                print(f"Tool Function 'search_news' found {len(simplified_articles)} articles.")
                return {"status": "success", "articles": simplified_articles}
            else:
                error_message = data.get("message", "Unknown API error")
                print(f"News API Error: {error_message}")
                return {"status": "error", "error_message": f"Failed to fetch news from NewsAPI: {error_message}"}
    except httpx.HTTPStatusError as e:
        error_text = f"HTTP error fetching news: {e.response.status_code}"
        try: error_text += f" - {e.response.text}"
        except Exception: pass
        print(error_text)
        return {"status": "error", "error_message": error_text}
    except httpx.RequestError as e:
        error_message = f"Network error fetching news: {e}"
        print(error_message)
        return {"status": "error", "error_message": error_message}
    except Exception as e:
        error_message = f"An unexpected error occurred in search_news function: {e}"
        print(f"{error_message}\n{traceback.format_exc()}") # Log traceback here too
        return {"status": "error", "error_message": error_message}

# --- FunctionTool instance ---
news_search_tool = FunctionTool(func=search_news)

# --- Agent Definition (unchanged) ---
news_agent = Agent(
    name="news_agent",
    model=LiteLlm("gemini-1.5-flash"),
    tools=[news_search_tool],
    description="Suggests relevant tourist or cultural news for a user's trip destination.",
    instruction=(
        # ... (agent instructions remain the same) ...
        "You are an assistant that finds relevant tourist or cultural news for a user's trip.\n"
        "1. You will be given a destination, travel dates, and a budget.\n"
        "2. **Use the `search_news` tool** to find news articles related to tourism or culture for the given destination and dates. Use the destination name possibly combined with terms like 'tourism', 'culture', 'events', 'exhibitions' as the query for the tool.\n"
        "3. **Check the 'status' key in the result from the `search_news` tool.**\n"
        "4. **If the status is 'success'**: Select 2-3 news items from the 'articles' list that are most relevant to a tourist interested in cultural events or local happenings. Extract the 'title' (use as 'name'), 'description', and 'url' (use as 'source') for each selected item.\n"
        "5. **If the status is 'error'**: Inform the user that you couldn't retrieve news and briefly mention the error if appropriate (e.g., 'information not available' or 'technical issue'). In this case, respond with an empty list for the 'news' key in the final JSON.\n"
        "6. **VERY IMPORTANT**: Respond ONLY in JSON format. The JSON object must have a single key named 'news' which contains a list of the selected news objects (if successful) or an empty list (if an error occurred or no relevant news was found). Each object in the list must have the keys 'name', 'description', and 'source'.\n"
        "   - Example Success Output: `{\"news\": [{\"name\": \"Title1\", \"description\": \"Desc1\", \"source\": \"url1\"}, ...]}`\n"
        "   - Example Error/No Results Output: `{\"news\": []}`\n"
        "7. Do not include any explanations or introductory text outside the final JSON structure."
    )
)

# --- Session Service and Runner Setup ---
# Explicitly note that InMemorySessionService is non-persistent.
print("Initializing InMemorySessionService (NOTE: State is NOT persistent across restarts)")
session_service = InMemorySessionService()
runner = Runner(
    agent=news_agent,
    app_name=APP_NAME_CONST,
    session_service=session_service
)

# --- Execute Function (API Endpoint Logic) ---
async def execute(request: dict[str, Any]):
    """
    Executes the news agent for a given travel request. Handles session creation.
    """
    current_user_id = USER_ID_CONST
    current_session_id = SESSION_ID_CONST
    current_app_name = APP_NAME_CONST
    session: Session | None = None # Type hint for session variable

    # --- Session Handling ---
    # Attempt to get the session. If it doesn't exist (e.g., after a restart), create it.
    try:
        session = session_service.get_session(
            app_name=current_app_name, user_id=current_user_id, session_id=current_session_id
        )
        # This log might appear even if the session is stale from before a restart
        print(f"Found session in service: {current_user_id}/{current_session_id}")
    except KeyError:
        print(f"Session not found in service, creating new session: {current_user_id}/{current_session_id}")
        try:
            # Create the session and store it in the variable 'session'
            session = session_service.create_session(
                app_name=current_app_name,
                user_id=current_user_id,
                session_id=current_session_id,
                state={} # Initialize with empty state
            )
            print(f"Successfully created session: {current_user_id}/{current_session_id}")
        except Exception as create_error:
             print(f"CRITICAL: Failed to create session: {create_error}\n{traceback.format_exc()}")
             return {"news": [], "error": f"Failed to initialize session: {create_error}"}
    except Exception as e:
        # Catch other potential errors during lookup
        print(f"Error during session lookup: {e}\n{traceback.format_exc()}")
        return {"news": [], "error": f"Session lookup error: {e}"}

    # --- Safety Check ---
    # Ensure we have a valid session object before proceeding
    if session is None:
         print("Error: Session object is None after get/create attempt.")
         return {"news": [], "error": "Session object unavailable."}

    # --- Construct Prompt ---
    prompt = (
        f"Please find tourist or cultural news based on the following trip details:\n"
        f"Destination: {request.get('destination', 'N/A')}\n"
        f"Start Date: {request.get('start_date', 'N/A')}\n"
        f"End Date: {request.get('end_date', 'N/A')}\n"
        f"Budget: {request.get('budget', 'N/A')}\n"
        f"Follow your instructions carefully."
    )
    message = types.Content(role="user", parts=[types.Part(text=prompt)])

    # --- Run Agent ---
    final_response_text = None
    print(f"\n--- Running Agent for Request: {request.get('destination', 'Unknown Destination')} (Session: {session.id}) ---")
    try:
        # The runner uses the session_id to look up the session in the session_service.
        # If the service state was cleared between the check above and now, this will fail.
        async for event in runner.run_async(user_id=current_user_id, session_id=current_session_id, new_message=message):

             print(f"Agent Event: {event.type}")
             if event.is_final_response():

                print("Final response received.")
                if event.content and event.content.parts:
                     final_response_text = event.content.parts[0].text
                     print(f"Raw Final Response Text:\n{final_response_text}")
                else:
                     print("Warning: Final response event has no content/parts.")
                     final_response_text = '{"news": []}'
                break
             elif event.is_error():
                 error_msg = f"Error Event during agent execution: {event.error_message}"
                 print(error_msg)
                 final_response_text = f'{{"news": [], "error": "{error_msg}"}}'
                 break

    except ValueError as ve:

        error_msg = (f"Caught ValueError during run_async: {ve}. "
                     f"This confirms the session '{current_session_id}' was not found by the runner, "
                     f"likely due to InMemorySessionService limitations (e.g., server restart clearing state) "
                     f"even if it seemed to exist moments before.")
        print(f"{error_msg}\n{traceback.format_exc()}")

        final_response_text = json.dumps({"news": [], "error": error_msg})
    except Exception as e:
        error_msg = f"Caught Exception during run_async: {e}"
        print(f"{error_msg}\n{traceback.format_exc()}")
        final_response_text = json.dumps({"news": [], "error": error_msg})


    # --- Process Final Response (unchanged from previous correct version) ---
    print("--- Processing Final Response ---")
    # ... (response cleaning and parsing logic remains the same) ...
    if final_response_text:
        try:
            cleaned_response_text = final_response_text.strip()
            if cleaned_response_text.startswith("```json"): cleaned_response_text = cleaned_response_text[7:]
            if cleaned_response_text.endswith("```"): cleaned_response_text = cleaned_response_text[:-3]
            cleaned_response_text = cleaned_response_text.strip()

            if not cleaned_response_text:
                 print("Warning: Cleaned response text is empty. Returning empty news list.")
                 return {"news": [], "warning": "Empty response from agent."}

            parsed = json.loads(cleaned_response_text)

            if isinstance(parsed, dict) and "error" in parsed:
                 print(f"Returning structured error: {parsed}")
                 if "news" not in parsed: parsed["news"] = []
                 return parsed

            if isinstance(parsed, dict) and "news" in parsed and isinstance(parsed["news"], list):
                 all_valid = True
                 for item in parsed["news"]:
                     if not (isinstance(item, dict) and "name" in item and "description" in item and "source" in item):
                         all_valid = False
                         print(f"Warning: Invalid item structure in news list: {item}")
                         break
                 if all_valid:
                     print("Parsed JSON response structure is valid.")
                     return {"news": parsed["news"]}
                 else:
                      error_msg = "Response JSON 'news' list contains invalid items."
                      print(f"Error: {error_msg}")
                      return {"news": [], "error": f"{error_msg} Raw: {cleaned_response_text}"}
            else:
                error_msg = "Response JSON is missing 'news' key or it's not a list."
                print(f"Error: {error_msg}")
                return {"news": [], "error": f"{error_msg} Raw: {cleaned_response_text}"}
        except json.JSONDecodeError as e:
            error_msg = f"JSON parsing failed: {e}"
            print(error_msg)
            print(f"Response content that failed parsing:\n{final_response_text}")
            return {"news": [], "error": f"{error_msg} Raw: {final_response_text}"}
    else:
         print("Error: No final response text was captured from the agent run.")
         return {"news": [], "error": "Agent did not produce a final response."}