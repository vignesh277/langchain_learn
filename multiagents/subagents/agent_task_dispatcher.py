from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent

# ==================================================
# Load Environment Variables
# ==================================================

load_dotenv()

# ==================================================
# Specialized Sub Agents
# ==================================================

research_agent = create_agent(
    model="groq:openai/gpt-oss-120b",
    system_prompt="You are a research specialist. Research topics deeply, provide factual information, and structure findings clearly."
)

writer_agent = create_agent(
    model="groq:openai/gpt-oss-120b",
    system_prompt="You are a writing specialist. Write high-quality, professional, well-formatted content."
)

# ==================================================
# Registry of Sub Agents
# ==================================================

SUBAGENTS = {
    "research": research_agent,
    "writer": writer_agent
}

# ==================================================
# Tool Used By Main Coordinator Agent
# ==================================================

@tool
def task(agent_name: str, description: str) -> str:
    """
    Launch a specialized subagent.

    Available agents:
    - research : Research and fact-finding
    - writer   : Content creation and editing
    """

    if agent_name not in SUBAGENTS:
        return f"Unknown agent: {agent_name}"

    agent = SUBAGENTS[agent_name]

    result = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": description
            }
        ]
    })

    return result["messages"][-1].content

# ==================================================
# Main Coordinator Agent
# ==================================================

main_agent = create_agent(
    model="groq:openai/gpt-oss-120b",
    tools=[task],
    system_prompt="You are a coordinator. Delegate to: research (fact-finding) or writer (content creation) using the task tool."
)

# ==================================================
# User Query
# ==================================================

query = """
Research the latest trends in AI agents
and then write a professional blog post.
"""

# ==================================================
# Execute Main Agent
# ==================================================

response = main_agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": query
        }
    ]
})

# ==================================================
# Print Final Response
# ==================================================

print("\n================ FINAL RESPONSE ================\n")

print(response["messages"][-1].content)
print(response["messages"])