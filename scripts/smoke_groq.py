"""Day 0 smoke test: call Groq once and print the reply."""
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()
llm = ChatGroq(model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"), temperature=0)
print(llm.invoke("Reply with exactly: telecom agent online").content)
