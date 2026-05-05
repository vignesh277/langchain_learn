from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage
from langchain_core.tools import tool

load_dotenv()


@tool
def divide(a: float, b: float) -> float:
    """Divide two numbers."""
    return a / b


@wrap_tool_call
def handle_tool_errors(request, handler):
    """Convert tool exceptions into a ToolMessage visible to the model."""
    try:
        return handler(request)
    except Exception as error:
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({error})",
            tool_call_id=request.tool_call["id"],
        )


def run_with_agent() -> None:
    agent = create_agent(
        model="groq:llama-3.1-8b-instant",
        tools=[divide],
        middleware=[handle_tool_errors],
        system_prompt="You must call the divide tool. Do not refuse or skip the tool call.",
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What is 10 divided by 0?"}]}
    )
    for message in result["messages"]:
        print(f"{type(message).__name__}: {message.content}")


def run_with_agent2() -> None:
    """Same as run_with_agent but WITHOUT middleware - will crash on tool error."""
    agent = create_agent(
        model="groq:llama-3.1-8b-instant",
        tools=[divide],
        system_prompt="You must call the divide tool. Do not refuse or skip the tool call.",
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What is 10 divided by 0?"}]}
    )
    for message in result["messages"]:
        print(f"{type(message).__name__}: {message.content}")


if __name__ == "__main__":
    print("=== WITH wrap_tool_call middleware ===")
    run_with_agent()
    print("\n=== WITHOUT wrap_tool_call middleware (will crash) ===")
    try:
        run_with_agent2()
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
