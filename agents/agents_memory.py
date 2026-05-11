from typing import Any

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage

# custom state ->class MemoryState(AgentState): → defines a new agent state type.
class MemoryState(AgentState):
    user_preferences: dict[str, str]
    user_name:str

# cutom middleware for the agent
class PreferenceMemoryMiddleware(AgentMiddleware):
    state_schema = MemoryState

    def before_model(
        self, state: MemoryState, runtime
    ) -> dict[str, Any] | None:
        """Store user preferences in short-term memory and feed them back to the model."""
        prefs = dict(state.get("user_preferences", {}))
        name=(state.get("user_name",""))
        latest = state["messages"][-1].content.lower()

        if "technical" in latest:
            prefs["style"] = "technical"
            prefs["verbosity"] = "detailed"
            return {
                "user_preferences": prefs,
                "messages": [
                    SystemMessage(
                        content=f"The user prefers technical, detailed explanations.and greet with {name}"
                    )
                
                ],
            }

        if "simple" in latest:
            prefs["style"] = "simple"
            prefs["verbosity"] = "short"
            return {
                "user_preferences": prefs,
                "messages": [
                    SystemMessage(content="The user prefers simple, short explanations.")
                ],
            }

        if prefs:
            return {
                "messages": [
                    SystemMessage(
                        content=(
                            f"Remember: the user prefers {prefs.get('style', 'plain')} "
                            f"and {prefs.get('verbosity', 'normal')} explanations."
                        )
                    )
                ]
            }

        return None


def main() -> None:
    load_dotenv()
    model = "groq:llama-3.1-8b-instant"

    middleware_agent = create_agent(
        model=model,
        middleware=[PreferenceMemoryMiddleware()],
    )

    first = middleware_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "I prefer technical explanations.",
                }
            ],
            "user_preferences": {},
            "user_name":"vignesh"
        }
    )

    # print("First turn:")
    # print(first)
    # print(first["messages"])
    # print("Stored state:", first["user_preferences"])

    second = middleware_agent.invoke(
        {
            "messages": first["messages"]
            + [{"role": "user", "content": "Explain memory in LangChain. and also greet "}],
            "user_preferences": first["user_preferences"],
            # "user_name":first["user_name"]
            
        }
    )

    print("\nSecond turn:")
    print(second["messages"][-1].content)
    print("Stored state:", second["user_preferences"])
# state schema only test creating a agent
    state_schema_agent = create_agent(
        model=model,
        state_schema=MemoryState,
    )
# state schema only test
    state_schema_result = state_schema_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Explain memory in LangChain.",
                }
            ],
            "user_preferences": {
                "style": "technical",
                "verbosity": "detailed",
            },
        }
    )

    # print("\nState schema only:")
    # print("Stored state:", state_schema_result["user_preferences"])
    # print(state_schema_result["messages"][-1].content)


if __name__ == "__main__":
    main()
