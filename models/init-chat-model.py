# Basic usage
# Models can be utilized in two ways:
# With agents - Models can be dynamically specified when creating an agent.
# Standalone - Models can be called directly (outside of the agent loop) 
# for tasks like text generation, classification, 
# or extraction without the need for an agent framework.


from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
 
load_dotenv()
 


model=init_chat_model("groq:llama-3.1-8b-instant",temperature=1,timeout=30,max_tokens=10,max_retries=6)

# Invoke 
# response = model.invoke("Why do parrots talk?")

# print(response)


# streaming 
# for chunk in model.stream("why do parrots have colorful feathers"):
#     print(chunk.text,end=" ",flush=True)

# full = None  # None | AIMessageChunk
# for chunk in model.stream("What color is the sky?"):
#     full = chunk if full is None else full + chunk
#     print(full.text)

# print(full.content_blocks)


# batch request 
# respones=model.batch([
#      "Why do parrots have colorful feathers?",
#     "How do airplanes fly?",
#     "What is quantum computing?"
# ]
# )

# for response in respones:
#     print(response.content)