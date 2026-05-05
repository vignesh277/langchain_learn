from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call,ModelRequest,ModelResponse

basicModel=ChatGroq(model="llama-3.1-8b-instant")

advancedModel=ChatGroq(model="llama-3.3-70b-versatile")

# dynamic model based on conditions

@wrap_model_call
def dynamic_model_selection(request:ModelRequest,handler)->ModelResponse:
    """choose model based on conversational complexity."""
    message_count=len(request.state["messages"])

    if message_count>1:
        model=advancedModel
    else:
        model=basicModel

    return handler(request.override(model=model))

agent=create_agent(model=basicModel,tools=[],middleware=[dynamic_model_selection])

result=agent.invoke({"messages":[{"role":"user","content":"what is your name"}]})

# result=agent.invoke({"messages":[{"role":"user","content":"what is your name"},{"role":"user","content":"which model are you and model name excatly are 3b instant or 70b verstaile "}]})

print(result["messages"][-1].response_metadata["model_name"])




