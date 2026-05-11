import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy
from pydantic import BaseModel


class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str


def pick_model() -> str:
    if os.getenv("GROQ_API_KEY"):
        return "groq:llama-3.1-8b-instant"
    raise RuntimeError("Set OPENAI_API_KEY or GROQ_API_KEY in your environment.")


def run_tool_strategy(model: str) -> None:
    agent = create_agent(
        model=model,
        response_format=ContactInfo,
    )
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Extract contact info: John Doe, john@example.com, (555) 123-4567",
                }
            ]
        }
    )
    print("\nToolStrategy output:")
    print(result)




def run_provider_strategy(model: str) -> None:
    """Explicitly use ProviderStrategy (native model structured output)."""
    agent = create_agent(
        model=model,
        response_format=ProviderStrategy(ContactInfo),
    )
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Extract contact info: Alice Johnson, alice@corp.com, (444) 555-6789",
                }
            ]
        }
    )
    print("\nProviderStrategy output (native model structured output):")
    print(result["structured_response"])


if __name__ == "__main__":
    load_dotenv()
    selected_model = pick_model()
    print(f"Model: {selected_model}")
    run_tool_strategy(selected_model)
    run_provider_strategy(selected_model)
