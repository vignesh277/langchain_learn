from dotenv import load_dotenv
load_dotenv()
from langchain.messages import HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool
from langchain.chat_models import init_chat_model

@tool
def get_weather(location:str)->str:
    """Get a current weather report for a location."""
    return f"The current weather in {location} is sunny and 72°F."

model = init_chat_model("groq:openai/gpt-oss-120b")

# Bind (potentially multiple) tools to the model
model_with_tools = model.bind_tools([get_weather])

# Step 1: Model generates tool calls
messages = [{"role": "user", "content": "What's the weather in Boston?"}]
print("STEP 1: sending messages to model")
print(messages)

ai_msg = model_with_tools.invoke(messages)
print("STEP 1: model response")
print(ai_msg)
print("STEP 1: tool calls")
print(ai_msg.tool_calls)

messages.append(ai_msg)
print("messages after appending model response")
print(messages)

# Step 2: Execute tools and collect results
for tool_call in ai_msg.tool_calls:
    # Execute the tool with the generated arguments
    print("STEP 2: executing tool call")
    print(tool_call)
    tool_result = get_weather.invoke(tool_call)
    print("STEP 2: tool result")
    print(tool_result)
    messages.append(tool_result)
    print("messages after appending tool result")
    print(messages)

# Step 3: Pass results back to model for final response
print("STEP 3: sending updated messages back to model")
print(messages)

final_response = model_with_tools.invoke(messages)
print("STEP 3: final response")
print(final_response)
print(final_response.content)
# "The current weather in Boston is 72°F and sunny."
