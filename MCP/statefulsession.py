import asyncio

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

load_dotenv()


async def main() -> None:
    client = MultiServerMCPClient(
        {
            "stateful": {
                "transport": "stdio",
                "command": "uv",
                "args": ["run", r"C:\Users\yenag\Desktop\New folder\MCP\stateful_server.py"],
            }
        }
    )

    async with client.session("stateful") as session:
        tools = await load_mcp_tools(session)

        agent = create_agent(
            model="groq:openai/gpt-oss-120b",
            tools=tools,
            system_prompt=(
                "You are a stateful memory assistant. Use the available MCP tools "
                "to store notes and manage the counter. Do not guess the stored "
                "value; always call the relevant tool. Return only the final answer."
            ),
        )

        print("Opened one persistent MCP session for stateful memory")

        store_response = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Store the note 'copilot session demo' and confirm it.",
                    }
                ]
            }
        )

        read_response = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "What note is stored right now?"}]}
        )

        counter_response_1 = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "Increase the counter by 1."}]}
        )

        counter_response_2 = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Increase the counter by 1 again, then tell me the counter.",
                    }
                ]
            }
        )

        def print_trace(label: str, response: dict) -> None:
            print(f"\n=== {label.upper()} RESPONSE ===")
            print(response["messages"][-1].content)
            print(f"--- {label.upper()} MESSAGE TRACE ---")
            for message in response["messages"]:
                print(f"{type(message).__name__}: {message.content}")
                if hasattr(message, "tool_calls") and message.tool_calls:
                    print(f"tool_calls={message.tool_calls}")

        print_trace("store note", store_response)
        print_trace("read note", read_response)
        print_trace("counter step 1", counter_response_1)
        print_trace("counter step 2", counter_response_2)


if __name__ == "__main__":
    asyncio.run(main())
