from dotenv import load_dotenv

load_dotenv()

from typing import Callable
from typing_extensions import NotRequired

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.tools import tool
from langchain_core.messages import HumanMessage


def _public_search_impl(query: str) -> str:
    """Shared implementation for public search aliases."""
    return f"Public results for: {query}"


@tool
def public_search(query: str) -> str:
    """Public search tool."""
    return _public_search_impl(query)


@tool
def brave_search(query: str) -> str:
    """Alias for search (some models try calling this name)."""
    return _public_search_impl(query)


@tool
def public_help(topic: str) -> str:
    """Public help tool."""
    return f"Help for: {topic}"


@tool
def advanced_search(query: str) -> str:
    """Advanced search tool."""
    return f"Advanced results for: {query}"


class AgentStateWithAuth(AgentState):
    """Custom graph state so `authenticated` is preserved from invoke → middleware."""

    authenticated: NotRequired[bool]


def select_allowed_tools(tools: list, is_authenticated: bool, messages_count: int) -> list:
    """Pure function to make filtering testable without live model calls."""
    if not is_authenticated:
        allowed = {"brave_search"}
        return [t for t in tools if t.name.startswith("public_") or t.name in allowed]

    if messages_count < 2:
        return [t for t in tools if t.name != "advanced_search"]

    return tools


@wrap_model_call
def state_based_tools(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Filter tools based on conversational state."""

    state = request.state
    is_authenticated = state.get("authenticated", False)
    messages = state.get("messages", [])
    messages_count = len(messages)

    # Debug: confirm what tools the model is allowed to call
    print("authenticated:", is_authenticated, "messages_count:", messages_count)
    print("tools_in_request:", [t.name for t in request.tools])

    tools = select_allowed_tools(request.tools, is_authenticated, messages_count)
    print("tools_selected:", [t.name for t in tools])
    request = request.override(tools=tools)

    return handler(request)


agent = create_agent(
    model="groq:llama-3.1-8b-instant",
    tools=[public_search, brave_search, public_help, advanced_search],
    middleware=[state_based_tools],
    state_schema=AgentStateWithAuth,
    system_prompt=(
        "Use at most one tool per user request when a tool is needed. "
        "After a tool returns, answer in plain text and do not call more tools."
    ),
)


def run_local_filter_tests() -> None:
    """Deterministic tests for filtering logic (no API calls)."""
    all_tools = [public_search, brave_search, public_help, advanced_search]
    cases = [
        ("UNAUTH", False, 1),
        ("AUTH_EARLY", True, 1),
        ("AUTH_LATE", True, 2),
    ]

    for label, is_authenticated, messages_count in cases:
        selected = select_allowed_tools(all_tools, is_authenticated, messages_count)
        print(f"{label}: {[t.name for t in selected]}")


def run_groq_examples() -> None:
    """Call Groq through the agent so middleware runs and prints tool filtering each model step."""
    # recursion_limit caps agent graph steps (model + tool rounds) to avoid runaway loops
    cfg = {"recursion_limit": 12}

    cases: list[tuple[str, dict]] = [
        (
            "UNAUTH (public tools only)",
            {
                "messages": [HumanMessage(content="Search for python tutorials briefly.")],
                "authenticated": False,
            },
        ),
        (
            "AUTH_EARLY (no advanced_search yet)",
            {
                "messages": [HumanMessage(content="Do an advanced search for LLM eval tips.")],
                "authenticated": True,
            },
        ),
        (
            "AUTH_LATE (advanced_search allowed — 2+ messages)",
            {
                "messages": [
                    HumanMessage(content="Hello."),
                    HumanMessage(content="Now do an advanced search for LLM eval tips."),
                ],
                "authenticated": True,
            },
        ),
    ]

    for title, state in cases:
        print("\n" + "=" * 60)
        print(title)
        print("=" * 60)
        try:
            out = agent.invoke(state, config=cfg)
            last = out["messages"][-1]
            content = getattr(last, "content", str(last))
            print("FINAL:", content)
        except Exception as exc:  # noqa: BLE001 — learning script: show any provider error
            err = str(exc)
            if "429" in err or "rate limit" in err.lower():
                print("Groq rate limit — wait ~10s and retry, or reduce cases.")
            else:
                print("Error:", exc)


if __name__ == "__main__":
    run_groq_examples()
