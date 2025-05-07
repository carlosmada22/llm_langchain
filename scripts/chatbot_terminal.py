
import os
import sys
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load model name from environment variable or fallback
model_name = os.getenv("LLM_MODEL", "llama3")
model = ChatOllama(model=model_name)

# Prompt template (customize this if you had specific roles or context)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{input}")
])

# Simple message trimming memory
conversation_history = []

MAX_HISTORY = 6  # Only keep the last 6 messages to control context length

def chat():
    print("Chatbot is ready! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == "exit":
            print("Goodbye!")
            break

        conversation_history.append(HumanMessage(content=user_input))
        if len(conversation_history) > MAX_HISTORY:
            conversation_history.pop(0)

        # Combine history with prompt
        messages = []
        for msg in conversation_history:
            if isinstance(msg, HumanMessage):
                messages.append(("user", msg.content))
            elif isinstance(msg, AIMessage):
                messages.append(("assistant", msg.content))

        full_prompt = ChatPromptTemplate.from_messages([("system", "You are a helpful assistant.")] + messages)
        chain = full_prompt | model | StrOutputParser()
        
        response = chain.invoke({"input": user_input})
        print("Bot:", response.strip())
        conversation_history.append(AIMessage(content=response.strip()))

if __name__ == "__main__":
    chat()
