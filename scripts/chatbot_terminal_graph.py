import getpass
import os
from typing import Sequence
from typing_extensions import Annotated, TypedDict

from langsmith import utils
utils.tracing_is_enabled()

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_core.messages import trim_messages
from dotenv import load_dotenv

# Setup model
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_TRACING_v2"] = "true"
os.environ["LANGSMITH_PROJECT"] = "tutorial_chatbot"
os.environ["LANGSMITH_API_KEY"] = ""


load_dotenv(dotenv_path="../.env", override=True)

model_name = os.getenv("LLM_MODEL") or "llama3.2"
model = ChatOllama(model=model_name)

# Prompt template with language control and placeholder for conversation
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer all questions to the best of your ability in {language}."),
    MessagesPlaceholder(variable_name="messages"),
])

# State definition
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str

# Define model call function for LangGraph
def call_model(state: State) -> dict:
    prompt = prompt_template.invoke(state)
    response = model.invoke(prompt)
    return {"messages": [response]}

# Graph setup
workflow = StateGraph(state_schema=State)
workflow.add_node("model", call_model)
workflow.set_entry_point("model")

# Memory saver for persistent session
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Message trimmer
trimmer = trim_messages(
    max_tokens=500,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

# CLI loop
def chat():
    print("Chatbot with LangGraph is ready! Type 'exit' to quit.")
    thread_id = "terminal_session_001"
    config = {"configurable": {"thread_id": thread_id}}
    language = input("Select a language (e.g., English, Spanish): ").strip() or "English"
    
    history: list[BaseMessage] = []
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        history.append(HumanMessage(content=user_input))

        trimmed = trimmer.invoke([
            SystemMessage(content="You are a helpful assistant."),
            *history
        ])
        
        inputs = {"messages": trimmed[1:], "language": language}
        output = app.invoke(inputs, config)
        response = output["messages"][-1]
        print("Bot:", response.content.strip())
        history.append(response)

if __name__ == "__main__":
    chat()
