# LLM, Tool - Google Search Tool, Agent, Memory, Streaming, Web Interface

from dotenv import load_dotenv
load_dotenv()
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langgraph.checkpoint.memory import MemorySaver  
import streamlit as st

llm =  ChatOllama(model="gemma4:31b-cloud", streaming=True)
search = GoogleSerperAPIWrapper()
tools = [search.run]

if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()
    st.session_state.history = []

agent = create_agent(
    model=llm,
    tools=tools,
    checkpointer=st.session_state.memory,
    system_prompt="You are a amazing AI agent and can search on Google as well"
)

# Building Web Interface
st.subheader("QuickAnswer - Answer at the speed of thought")

for message in st.session_state.history:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)

query = st.chat_input("Ask Anything...")
if query:
    st.chat_message("User").markdown(query)
    st.session_state.history.append({"role": "user", "content": query})    
    response = agent.invoke(
        {"messages":[{"role": "user", "content": query}]},
        {"configurable": {"thread_id":"asd123"}}
    )

    answer = response["messages"][-1].content
    st.chat_message("AI").markdown(answer)
    st.session_state.history.append({"role": "AI", "content": answer})   