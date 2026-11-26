"""Day 0 smoke test: call the Gemini backup once and print the reply."""
import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0,
)
print(llm.invoke("Reply with exactly: backup agent online").content)
