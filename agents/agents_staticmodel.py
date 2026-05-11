from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain_groq import ChatGroq

# agent=create_agent("groq:llama-3.1-8b-instant",tools=tools)

# static model
model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0,
    max_retries=2,
    max_tokens=1000,
    #other parameters 
)


agent=create_agent(model,tools=[],)

result=agent.invoke(
    {"messages":[{"role":"user","content":"hello"}]}
)

print(result["messages"][-1].content)

