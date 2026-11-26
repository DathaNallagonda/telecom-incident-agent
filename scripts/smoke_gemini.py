"""Day 0 smoke test: call the Gemini backup once and print the reply.

Tries GEMINI_MODEL first, then a lighter model, since Flash models
sometimes return 503 under high demand.
"""

import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
for model in [os.getenv("GEMINI_MODEL", "gemini-3.8-flash"), "gemini-3.5-flash-lite"]:
    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0,
        timeout=30,
        max_retries=1,
    )
    try:
        print(f"{model}: {llm.invoke('Reply with exactly: backup agent online').text}")
        break
    except Exception as e:
        print(f"{model}: failed ({type(e).__name__}: {str(e)[:80]})")
