import asyncio
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

load_dotenv()

async def main():
    client = MultiServerMCPClient(
        {
            "math": {
                "transport": "stdio",
                "command": "uv",
                "args": ["run", r"C:\Users\yenag\Desktop\New folder\MCP\custom_server.py"],
            },

            "weather": {
                "transport": "http",
                "url":"http://127.0.0.1:8000/mcp",
            }
        } # pyright: ignore[reportArgumentType]
    )

    tools = await client.get_tools()
    system_prompt = (
        "You are a multi-tool assistant with access to multiple MCP servers. "
        "Inspect the user's question and choose the relevant tool or tools. "
        "If the question can be answered by more than one server, call each needed tool in sequence and combine the results. "
        "Do not guess when a relevant tool is available. Return only the final answer."
    )

    agent = create_agent(
        model="groq:openai/gpt-oss-120b",
        tools=tools,
        system_prompt=system_prompt,
    )

    math_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "what's (3 + 5) x 12?"}]}
    )
    weather_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "what is the weather in Hyderabad?"}]}
    )

    def print_trace(label: str, response: dict) -> None:
        print(f"\n=== {label.upper()} RESPONSE ===")
        print(response)
        print(f"--- {label.upper()} MESSAGE TRACE ---")
        for message in response["messages"]:
            print(f"{type(message).__name__}: {message.content}")
            if hasattr(message, "tool_calls") and message.tool_calls:
                print(f"tool_calls={message.tool_calls}")

    print_trace("math", math_response)
    print_trace("weather", weather_response)

if __name__ == "__main__":
    asyncio.run(main())

