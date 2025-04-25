from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, max_tokens=1000, api_key=GEMINI_API_KEY)

response = llm.invoke("What is the capital of France?")
print(response.content)  