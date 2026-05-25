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

# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

# =========================================================
# DEBUG LOGGER
# =========================================================

def debug_log(title: str):

    """
    Pretty debug logger.
    """

    print("\n" + "=" * 80)
    print(f"🔥 DEBUG: {title}")
    print("=" * 80)


# =========================================================
# SHARED STATE
# =========================================================

class AgentState(TypedDict):

    """
    Shared state across entire graph.
    """

    # add_messages automatically appends messages
    messages: Annotated[
        list,
        add_messages
    ]

    active_agent: str


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

    Args:
        reason: Reason for transfer

    Returns:
        Command object updating active agent
    """

    debug_log("TOOL EXECUTION: transfer_to_support")

    print(f"Reason Received: {reason}")

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

    Args:
        reason: Reason for transfer

    Returns:
        Command object updating active agent
    """

    debug_log("TOOL EXECUTION: transfer_to_sales")

    print(f"Reason Received: {reason}")

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
    Record customer warranty status.

    Args:
        status: Warranty status

    Returns:
        Warranty confirmation message
    """

    debug_log("TOOL EXECUTION: record_warranty")

    print(f"Warranty Status: {status}")

    result = f"Warranty recorded: {status}"

    print(f"Tool Result: {result}")

    return result


@tool
def provide_support(solution: str) -> str:
    """
    Provide technical troubleshooting support.

    Args:
        solution: Suggested troubleshooting solution

    Returns:
        Support response message
    """

    debug_log("TOOL EXECUTION: provide_support")

    print(f"Support Solution: {solution}")

    result = f"Support solution: {solution}"

    print(f"Tool Result: {result}")

    return result


# =========================================================
# SALES TOOLS
# =========================================================

@tool
def recommend_product(product: str) -> str:
    """
    Recommend a product to the customer.

    Args:
        product: Product recommendation

    Returns:
        Product recommendation message
    """

    debug_log("TOOL EXECUTION: recommend_product")

    print(f"Recommended Product: {product}")

    result = f"Recommended product: {product}"

    print(f"Tool Result: {result}")

    return result


@tool
def pricing_info(price: str) -> str:
    """
    Provide pricing information.

    Args:
        price: Product pricing details

    Returns:
        Pricing response message
    """

    debug_log("TOOL EXECUTION: pricing_info")

    print(f"Pricing Info: {price}")

    result = f"Pricing details: {price}"

    print(f"Tool Result: {result}")

    return result


# =========================================================
# COORDINATOR NODE
# =========================================================

def coordinator_node(state: AgentState):

    """
    Main supervisor agent.
    """

    debug_log("COORDINATOR NODE START")

    print(f"Current Active Agent: {state['active_agent']}")

    print("\nCURRENT MESSAGES:\n")

    for msg in state["messages"]:

        print(
            f"{type(msg).__name__}: "
            f"{msg.content}"
        )

    system_prompt = middleware(state)

    print("\nSYSTEM PROMPT:\n")
    print(system_prompt)

    model = llm.bind_tools([

        transfer_to_support,
        transfer_to_sales

    ])

    print("\nCalling Coordinator LLM...\n")

    response = model.invoke([

        SystemMessage(content=system_prompt),

        *state["messages"]

    ])

    print("\nCOORDINATOR RESPONSE:\n")

    print(response)

    if response.tool_calls:

        print("\nTOOL CALLS DETECTED:\n")

        for tool_call in response.tool_calls:

            print(f"Tool Name: {tool_call['name']}")
            print(f"Arguments: {tool_call['args']}")

    debug_log("COORDINATOR NODE END")

    return {
        "messages": [response]
    }


# =========================================================
# SUPPORT NODE
# =========================================================

def support_node(state: AgentState):

    """
    Technical support agent.
    """

    debug_log("SUPPORT NODE START")

    print(f"Current Active Agent: {state['active_agent']}")

    print("\nCURRENT MESSAGES:\n")

    for msg in state["messages"]:

        print(
            f"{type(msg).__name__}: "
            f"{msg.content}"
        )

    system_prompt = middleware(state)

    print("\nSYSTEM PROMPT:\n")
    print(system_prompt)

    model = llm.bind_tools([

        record_warranty,
        provide_support

    ])

    print("\nCalling Support LLM...\n")

    response = model.invoke([

        SystemMessage(content=system_prompt),

        *state["messages"]

    ])

    print("\nSUPPORT RESPONSE:\n")

    print(response)

    if response.tool_calls:

        print("\nSUPPORT TOOL CALLS:\n")

        for tool_call in response.tool_calls:

            print(f"Tool Name: {tool_call['name']}")
            print(f"Arguments: {tool_call['args']}")

    debug_log("SUPPORT NODE END")

    return {
        "messages": [response]
    }


# =========================================================
# SALES NODE
# =========================================================

def sales_node(state: AgentState):

    """
    Sales department agent.
    """

    debug_log("SALES NODE START")

    print(f"Current Active Agent: {state['active_agent']}")

    print("\nCURRENT MESSAGES:\n")

    for msg in state["messages"]:

        print(
            f"{type(msg).__name__}: "
            f"{msg.content}"
        )

    system_prompt = middleware(state)

    print("\nSYSTEM PROMPT:\n")
    print(system_prompt)

    model = llm.bind_tools([

        recommend_product,
        pricing_info

    ])

    print("\nCalling Sales LLM...\n")

    response = model.invoke([

        SystemMessage(content=system_prompt),

        *state["messages"]

    ])

    print("\nSALES RESPONSE:\n")

    print(response)

    if response.tool_calls:

        print("\nSALES TOOL CALLS:\n")

        for tool_call in response.tool_calls:

            print(f"Tool Name: {tool_call['name']}")
            print(f"Arguments: {tool_call['args']}")

    debug_log("SALES NODE END")

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

    """
    Route coordinator output.
    """

    debug_log("MAIN ROUTER")

    last_message = state["messages"][-1]

    print(
        "Last Message Type:",
        type(last_message).__name__
    )

    if isinstance(last_message, AIMessage):

        if last_message.tool_calls:

            tool_name = (
                last_message.tool_calls[0]["name"]
            )

            print(f"Detected Tool: {tool_name}")

            print("Routing to main_tools")

            return "main_tools"

    print("Routing to END")

    return END


# =========================================================
# AFTER MAIN TOOLS ROUTER
# =========================================================

def after_main_tools(state: AgentState):

    """
    Route after transfer tools execute.
    """

    debug_log("AFTER MAIN TOOLS ROUTER")

    active_agent = state.get("active_agent")

    print(f"Updated Active Agent: {active_agent}")

    if active_agent == "support":

        print("Routing to SUPPORT SUBGRAPH")

        return "support_subgraph"

    if active_agent == "sales":

        print("Routing to SALES SUBGRAPH")

        return "sales_subgraph"

    print("Routing to END")

    return END


# =========================================================
# SUPPORT ROUTER
# =========================================================

def router_support_tools(state: AgentState):

    """
    Route support agent output.
    """

    debug_log("SUPPORT ROUTER")

    last_message = state["messages"][-1]

    print(
        "Last Message Type:",
        type(last_message).__name__
    )

    if isinstance(last_message, AIMessage):

        if last_message.tool_calls:

            print("Routing to support_tools")

            return "support_tools"

    print("Routing to END")

    return END


# =========================================================
# SALES ROUTER
# =========================================================

def router_sales_tools(state: AgentState):

    """
    Route sales agent output.
    """

    debug_log("SALES ROUTER")

    last_message = state["messages"][-1]

    print(
        "Last Message Type:",
        type(last_message).__name__
    )

    if isinstance(last_message, AIMessage):

        if last_message.tool_calls:

            print("Routing to sales_tools")

            return "sales_tools"

    print("Routing to END")

    return END


# =========================================================
# SUPPORT SUBGRAPH
# =========================================================

def build_support_subgraph():

    """
    Build support workflow subgraph.
    """

    debug_log("BUILDING SUPPORT SUBGRAPH")

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

    graph.add_edge(
        "support_tools",
        "support_agent"
    )

    return graph.compile()


# =========================================================
# SALES SUBGRAPH
# =========================================================

def build_sales_subgraph():

    """
    Build sales workflow subgraph.
    """

    debug_log("BUILDING SALES SUBGRAPH")

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

    graph.add_edge(
        "sales_tools",
        "sales_agent"
    )

    return graph.compile()


# =========================================================
# MAIN GRAPH
# =========================================================

def build_graph():

    """
    Build complete workflow graph.
    """

    debug_log("BUILDING MAIN GRAPH")

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

    debug_log("GRAPH BUILD COMPLETE")

    return workflow.compile()


# =========================================================
# MAIN
# =========================================================

def main():

    """
    Application entry point.
    """

    debug_log("APPLICATION START")

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

        "active_agent": "coordinator"
    }

    debug_log("INITIAL STATE")

    print(initial_state)

    print("\nSTARTING GRAPH EXECUTION...\n")

    # =====================================================
    # STREAM EXECUTION
    # =====================================================

    for chunk in graph.stream(

        initial_state,

        config={
            "recursion_limit": 20
        }

    ):

        debug_log("GRAPH STREAM EVENT")

        print(chunk)

    debug_log("APPLICATION END")


# =========================================================
# ENTRY
# =========================================================

if __name__ == "__main__":

    main()