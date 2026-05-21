from dotenv import load_dotenv
from typing_extensions import TypedDict, Annotated

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage
)

from langchain_core.tools import (
    tool,
    InjectedToolCallId
)

from langchain_groq import ChatGroq

from langgraph.graph import (
    StateGraph,
    START,
    END
)

from langgraph.graph.message import add_messages

from langgraph.prebuilt import ToolNode
from langgraph.types import Command

load_dotenv()

# =========================================================
# SHARED STATE
# =========================================================

class AgentState(TypedDict):

    # add_messages automatically appends messages
    messages: Annotated[
        list,
        add_messages
    ]

    active_agent: str

    support_completed: bool

    sales_completed: bool


# =========================================================
# MODEL
# =========================================================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)


# =========================================================
# MIDDLEWARE
# =========================================================

def middleware(state: AgentState):

    """
    Dynamic middleware layer.

    Changes prompts/configuration
    based on current workflow state.
    """

    active_agent = state.get(
        "active_agent",
        "coordinator"
    )

    prompts = {

        "coordinator":
        """
        You are a coordinator agent.

        Responsibilities:
        - Understand user intent
        - Route user to correct department
        - Use ONLY transfer tools

        Rules:
        - Warranty/display/technical issues
          -> transfer_to_support

        - Pricing/purchase/product recommendation
          -> transfer_to_sales

        IMPORTANT:
        - NEVER answer directly
        - ALWAYS use transfer tools
        """,

        "support":
        """
        You are a support specialist.

        Responsibilities:
        - Handle warranty issues
        - Troubleshoot technical problems
        - Help users resolve issues

        Workflow:
        1. Record warranty
        2. Provide support solution
        3. Then provide final answer

        IMPORTANT:
        - Use tools only when necessary
        - Do NOT repeatedly call tools
        """,

        "sales":
        """
        You are a sales specialist.

        Responsibilities:
        - Handle pricing
        - Recommend products
        - Help customers purchase

        Workflow:
        1. Recommend product
        2. Provide pricing
        3. Then provide final answer

        IMPORTANT:
        - Do NOT repeatedly call tools
        """
    }

    return prompts.get(
        active_agent,
        prompts["coordinator"]
    )


# =========================================================
# HANDOFF TOOLS
# =========================================================

@tool
def transfer_to_support(
    reason: str,
    tool_call_id: Annotated[
        str,
        InjectedToolCallId
    ]
) -> Command:

    """
    Transfer conversation to support department.
    """

    return Command(
        update={

            "active_agent": "support",

            "messages": [

                ToolMessage(
                    content=(
                        "Successfully transferred "
                        "to support department."
                    ),
                    tool_call_id=tool_call_id
                )
            ]
        }
    )


@tool
def transfer_to_sales(
    reason: str,
    tool_call_id: Annotated[
        str,
        InjectedToolCallId
    ]
) -> Command:

    """
    Transfer conversation to sales department.
    """

    return Command(
        update={

            "active_agent": "sales",

            "messages": [

                ToolMessage(
                    content=(
                        "Successfully transferred "
                        "to sales department."
                    ),
                    tool_call_id=tool_call_id
                )
            ]
        }
    )


# =========================================================
# SUPPORT TOOLS
# =========================================================

@tool
def record_warranty(status: str) -> str:

    """
    Record warranty status.
    """

    return f"Warranty recorded: {status}"


@tool
def provide_support(solution: str) -> str:

    """
    Provide support solution.
    """

    return f"Support solution: {solution}"


# =========================================================
# SALES TOOLS
# =========================================================

@tool
def recommend_product(product: str) -> str:

    """
    Recommend a product.
    """

    return f"Recommended product: {product}"


@tool
def pricing_info(price: str) -> str:

    """
    Provide pricing information.
    """

    return f"Pricing details: {price}"


# =========================================================
# COORDINATOR NODE
# =========================================================

def coordinator_node(state: AgentState):

    system_prompt = middleware(state)

    model = llm.bind_tools([

        transfer_to_support,
        transfer_to_sales

    ])

    response = model.invoke([

        SystemMessage(content=system_prompt),

        *state["messages"]

    ])

    return {
        "messages": [response]
    }


# =========================================================
# SUPPORT NODE
# =========================================================

def support_node(state: AgentState):

    system_prompt = middleware(state)

    model = llm.bind_tools([

        record_warranty,
        provide_support

    ])

    response = model.invoke([

        SystemMessage(content=system_prompt),

        *state["messages"]

    ])

    return {
        "messages": [response]
    }


# =========================================================
# SALES NODE
# =========================================================

def sales_node(state: AgentState):

    system_prompt = middleware(state)

    model = llm.bind_tools([

        recommend_product,
        pricing_info

    ])

    response = model.invoke([

        SystemMessage(content=system_prompt),

        *state["messages"]

    ])

    return {
        "messages": [response]
    }


# =========================================================
# TOOL NODES
# =========================================================

main_tool_node = ToolNode([

    transfer_to_support,
    transfer_to_sales

])

support_tool_node = ToolNode([

    record_warranty,
    provide_support

])

sales_tool_node = ToolNode([

    recommend_product,
    pricing_info

])


# =========================================================
# MAIN ROUTER
# =========================================================

def route_main(state: AgentState):

    last_message = state["messages"][-1]

    if isinstance(last_message, AIMessage):

        if last_message.tool_calls:
            return "main_tools"

    return END


# =========================================================
# AFTER MAIN TOOLS ROUTER
# =========================================================

def after_main_tools(state: AgentState):

    active_agent = state.get("active_agent")

    if active_agent == "support":
        return "support_subgraph"

    if active_agent == "sales":
        return "sales_subgraph"

    return END


# =========================================================
# SUPPORT ROUTER
# =========================================================

def router_support_tools(state: AgentState):

    last_message = state["messages"][-1]

    if isinstance(last_message, AIMessage):

        if last_message.tool_calls:
            return "support_tools"

    return END


# =========================================================
# SALES ROUTER
# =========================================================

def router_sales_tools(state: AgentState):

    last_message = state["messages"][-1]

    if isinstance(last_message, AIMessage):

        if last_message.tool_calls:
            return "sales_tools"

    return END


# =========================================================
# SUPPORT SUBGRAPH
# =========================================================

def build_support_subgraph():

    graph = StateGraph(AgentState)

    graph.add_node(
        "support_agent",
        support_node
    )

    graph.add_node(
        "support_tools",
        support_tool_node
    )

    graph.add_edge(
        START,
        "support_agent"
    )

    graph.add_conditional_edges(
        "support_agent",
        router_support_tools
    )

    # AFTER TOOL EXECUTION
    # GO BACK TO AGENT

    graph.add_edge(
        "support_tools",
        "support_agent"
    )

    return graph.compile()


# =========================================================
# SALES SUBGRAPH
# =========================================================

def build_sales_subgraph():

    graph = StateGraph(AgentState)

    graph.add_node(
        "sales_agent",
        sales_node
    )

    graph.add_node(
        "sales_tools",
        sales_tool_node
    )

    graph.add_edge(
        START,
        "sales_agent"
    )

    graph.add_conditional_edges(
        "sales_agent",
        router_sales_tools
    )

    # AFTER TOOL EXECUTION
    # GO BACK TO AGENT

    graph.add_edge(
        "sales_tools",
        "sales_agent"
    )

    return graph.compile()


# =========================================================
# MAIN GRAPH
# =========================================================

def build_graph():

    workflow = StateGraph(AgentState)

    support_subgraph = (
        build_support_subgraph()
    )

    sales_subgraph = (
        build_sales_subgraph()
    )

    # =====================================================
    # NODES
    # =====================================================

    workflow.add_node(
        "coordinator",
        coordinator_node
    )

    workflow.add_node(
        "main_tools",
        main_tool_node
    )

    workflow.add_node(
        "support_subgraph",
        support_subgraph
    )

    workflow.add_node(
        "sales_subgraph",
        sales_subgraph
    )

    # =====================================================
    # FLOW
    # =====================================================

    workflow.add_edge(
        START,
        "coordinator"
    )

    workflow.add_conditional_edges(
        "coordinator",
        route_main
    )

    workflow.add_conditional_edges(
        "main_tools",
        after_main_tools
    )

    workflow.add_edge(
        "support_subgraph",
        END
    )

    workflow.add_edge(
        "sales_subgraph",
        END
    )

    return workflow.compile()


# =========================================================
# MAIN
# =========================================================

def main():

    graph = build_graph()

    initial_state = {

        "messages": [

            HumanMessage(
                content=(
                    "My laptop has warranty "
                    "and has display issues"
                )
            )
        ],

        "active_agent": "coordinator",

        "support_completed": False,

        "sales_completed": False
    }

    result = graph.invoke(

        initial_state,

        config={
            "recursion_limit": 20
        }
    )

    print(
        "\n================ RESULT ================\n"
    )

    print(
        "\n========== DETAILED ROUTING RESULT ==========\n"
    )

    for i, msg in enumerate(result["messages"]):

        msg_type = type(msg).__name__

        print(f"[{i}] {msg_type}:")

        # =================================================
        # CONTENT
        # =================================================

        if hasattr(msg, "content"):

            if str(msg.content).strip():

                print(
                    f"  Content: {msg.content}"
                )

        # =================================================
        # TOOL CALLS
        # =================================================

        if hasattr(msg, "tool_calls"):

            if msg.tool_calls:

                for call in msg.tool_calls:

                    print(
                        f"  🛠️ Tool Call Triggered: "
                        f"'{call['name']}'"
                    )

                    print(
                        f"    Arguments: "
                        f"{call['args']}"
                    )

        print("-" * 50)


# =========================================================
# ENTRY
# =========================================================

if __name__ == "__main__":
    main()