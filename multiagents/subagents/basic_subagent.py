from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict,Literal

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage,AIMessage

from langgraph.graph import StateGraph,END
from langgraph.types import Command

# LLM

llm=ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# STATE

class AgentState(TypedDict):
    messages:list



# Research Agent

def research_agent(state:AgentState):

    messages=state["messages"]
    print("\n🔍 RESEARCH AGENT CALLED")
    print(f"   Input Query: {messages[-1].content}")

    response=llm.invoke([
        HumanMessage(
            content=f"""
            You are a research specialist.
            
            Explain deeply:
            {messages[-1].content}
            """
        )
    ])

    print(f"   Response Length: {len(response.content)} characters")
    print(f"   Response Preview: {response.content[:100]}...")

    return {
        "messages":messages+[
            AIMessage(content=response.content)
        ]
    }


# coding agent

def coding_agent(state:AgentState):
    
    messages=state["messages"]
    print("\n💻 CODING AGENT CALLED")
    print(f"   Input Query: {messages[-1].content}")

    response=llm.invoke([
        HumanMessage(
            content=f"""
            You are an expert python enginner

            write clean production-ready code for

            {messages[-1].content}
            """
        )
    ])

    print(f"   Response Length: {len(response.content)} characters")
    print(f"   Response Preview: {response.content[:100]}...")

    return {
        "messages":messages+[
            AIMessage(content=response.content)
        ]
    }

# supervisior


def supervisior(state:AgentState) -> Command[ Literal["research_agent","coding_agent"]]:

    query=state["messages"][-1].content.lower()
    print("\n🚦 SUPERVISOR ROUTING")
    print(f"   Original Query: {state['messages'][-1].content}")
    print(f"   Query (lowercase): {query}")
    # Simple rule-based routing.
    # Real Systems Use Usually:LLM routing
    # semantic routing
    # embedding routing
    # classifier routing
    if "code" in query or "python" in query:
        print(f"   ✅ Routing Decision: CODING_AGENT (detected 'code' or 'python')")
        return Command(goto="coding_agent")

    print(f"   ✅ Routing Decision: RESEARCH_AGENT (default)")
    return Command(goto="research_agent")


# graph

builder=StateGraph(AgentState)

builder.add_node("supervisor",supervisior)
builder.add_node("research_agent",research_agent)
builder.add_node("coding_agent",coding_agent)

builder.set_entry_point("supervisor")

builder.add_edge("research_agent",END)
builder.add_edge("coding_agent",END)

graph=builder.compile()


# RUn

print("=" * 80)
print("🤖 STARTING MULTI-AGENT GRAPH EXECUTION")
print("=" * 80)

user_query = "Write python code for binary search"
print(f"\n📝 User Query: {user_query}")

result=graph.invoke({
    "messages":[
        HumanMessage(
            content=user_query
        )
    ]
})

print("\n" + "=" * 80)
print("📤 FINAL OUTPUT")
print("=" * 80)
print(f"\nTotal messages in conversation: {len(result['messages'])}")
print(f"\nAgent Response:\n{result['messages'][-1].content}")
print("\n" + "=" * 80)

