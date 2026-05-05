from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model

model=init_chat_model("groq:openai/gpt-oss-120b")
# streaming reasoning output
# for chunk in model.stream("Why do parrots have colorful feathers?"):
#     reasoning_steps = [r for r in chunk.content_blocks if r["type"] == "reasoning"]
#     print(reasoning_steps if reasoning_steps else chunk.text)

# complete reasoning output

response=model.invoke("Why do parrots have colorful feathers?")
reasoning_steps=[b for b in response.content_blocks if b["type"]=="reasoning"]
print(" ".join(step["reasoning"]for step in reasoning_steps))