import os
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing_extensions import Annotated, TypedDict
from typing import Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


#init models
model = init_chat_model(model="gemini-2.0-flash", model_provider="google_genai")

# needed for the custom prompt template to add custom variables to the state
class MessagesState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str


# define the prompt template
prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant.You should always address the user as 'buddy' and always use the {language} language",),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Define a new graph for memory
# The graph will have a single node that calls the model
workflow = StateGraph(state_schema=MessagesState)


# Define the function that calls the model
def call_model(state: MessagesState):
    prompt = prompt_template.invoke(state)  # Get the prompt from the state
    response = model.invoke(prompt)
    return {"messages": response}


# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "abc123"}} #change this to desired thread id

while True:
    query = input("User: ")
    if query.lower() == "exit":
        print("Exiting!!")
        break
    input_message = HumanMessage(query) 
    output = app.invoke(
        {"messages": input_message, "language" : "tamil"},config=config)
    output["messages"][-1].pretty_print()
