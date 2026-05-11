from typing import TypedDict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt ,ModelRequest




load_dotenv()


class Context(TypedDict):
    user_role:str

@dynamic_prompt
def user_role_prompt(request:ModelRequest) -> str:
    """Generate system prompt based on user role"""
    user_role=request.runtime.context.get("user_role","user")
    base_prompt="you are a helpful assisstant"

    if user_role == "expert":
        print("expert system prompt")
        return f"{base_prompt} provide detailed technical responses"

    elif user_role=="beginner":
        print("beginner system prompt")
        return f"{base_prompt} explain concepts simply and avoid jargon"
    
    return base_prompt

agent = create_agent(
    model="groq:llama-3.1-8b-instant",
    # tools=[web_search],
    middleware=[user_role_prompt],
    context_schema=Context
    # name="research_assistant"  optinal name for agent when using multi agents systems
)

runtime_context: Context = {"user_role": "beginner"}

result= agent.invoke(
    {"messages":[{"role":"user","content":"explain machine learning "}]},
    context=runtime_context
)   

print(result["messages"][-1].content)

# literary_agent = create_agent (
#     model="groq:llama-3.1-8b-instant",
#     system_prompt=SystemMessage(
#         content=[
#             {
#                 "type":"text",
#                 "text":"you are an AI assisstant tasked with analyzing literacy works"
#                 ,
#             },

#             {
#                 "type":"text",
#                 "text":"<the entire contents of 'Pride and Prejudice'>",
#                 "cache_control": {"type": "ephemeral"}
#             }
#         ]
#     )
# )

# result=literary_agent.invoke(
#     {"messages":[HumanMessage("Analzye he major themes in 'Pride and Prejudice'.")]}
# )


# print(result["messages"][-1].content)



