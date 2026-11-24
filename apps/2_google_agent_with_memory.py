from dotenv import load_dotenv
load_dotenv()
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langgraph.checkpoint.memory import MemorySaver  

llm =  ChatOllama(model="gemma4:31b-cloud")
search = GoogleSerperAPIWrapper()
memory = MemorySaver()

agent = create_agent(
    model=llm,
    tools=[search.run],
    checkpointer=MemorySaver(),
    system_prompt="You are a helpful assistant that can answer questions and perform Google searches when needed."
)

while True:
    query = input("User: ")
    if query.lower() == 'exit':
        print("Exiting the agent. Goodbye!")
        break

    response = agent.invoke(
            {"messages":[{"role": "user", "content": query}]},
            {"configurable": {"thread_id":"asd123"}}
    )
    print("AI: ", response["messages"][-1].content)