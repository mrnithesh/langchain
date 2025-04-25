from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


llm = init_chat_model(model="gemini-2.0-flash",model_provider="google_genai")
prompt = ChatPromptTemplate.from_template("Write the story based on the user input!  User: {user}")
chain = prompt | llm | StrOutputParser()


def chat(user_input):
    ai_response = chain.invoke({"user": user_input})
    return ai_response

print("Enter exit to end the chat")

while True:
    user_input=input("User :")
    if user_input.lower() == "exit":
        print("Exiting!!")
        break
    print(chat(user_input))