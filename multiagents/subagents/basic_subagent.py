from langchain.tools import tool 
from langchain.agents import create_agent
from dotenv import load_dotenv
load_dotenv()
# create a subagent

subagent=create_agent(model="groq:openai/gpt-oss-120b")

# wrap it as a tool

@tool("research",description="Research a topic and find findings")
def call_research_agent(query:str):
    result=subagent.invoke({"messages":[{"role":"user","content":query}]})
    return result["messages"][-1].content

# main agent with subagent as a tool
main_agent=create_agent(model="groq:openai/gpt-oss-120b",tools=[call_research_agent])


result=main_agent.invoke({"messages":[{"role":"user","content":"what is AI?"}]})

print(result["messages"][-1].content)
