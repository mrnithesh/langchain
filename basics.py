from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, max_tokens=1000, api_key=GEMINI_API_KEY)

def chat(user_input):
    ai_response = llm.invoke(HumanMessage(content = user_input))
    return ai_response.content

print("Enter exit to end the chat")

while True:
    user_input=input("User :")
    if user_input.lower() == "exit":
        print("Exiting!!")
        break
    print(chat(user_input))