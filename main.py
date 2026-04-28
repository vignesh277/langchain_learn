
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
load_dotenv()




agent =create_agent(
    model="groq:llama-3.1-8b-instant",
    system_prompt="You are a helpful assistant.",
)

result=agent.invoke(
    {"messages": [{"role": "user", "content": "what is your model name"}]}
)

print(result["messages"][-1].content)



