from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

@tool
def search(query:str)->str:
    """search for information"""
    return f"Results for :{query}"

@tool
def get_weather(location:str)->str:
    """Get weather information for a location."""
    return f"weather for :{location}:sunny,72 c"

model=ChatGroq(model="llama-3.3-70b-versatile")

agent=create_agent(model=model,tools=[search,get_weather])

result = agent.invoke({
    "messages": [HumanMessage(content="what is weather in SF?")]
})

print(result["messages"][-1].content)