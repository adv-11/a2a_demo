from common.a2a_server import create_app
from task_manager import run

import asyncio
import json
import os
import ssl
import urllib3

# Disable SSL verification warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure SSL context to disable verification (for environments with self-signed certs)
ssl._create_default_https_context = ssl._create_unverified_context

# Set environment variable to disable certificate verification
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['NODE_TLS_REJECT_UNAUTHORIZED'] = '0'

# Disable tracing for cleaner logs



app = create_app(agent=type("Agent", (), {"execute": run}))
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8001)