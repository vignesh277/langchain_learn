from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict, Callable

# =========================================================
# DEBUG LOGGER
# =========================================================

def debug_log(title: str, detail: str | None = None) -> None:
    """Pretty debug logger for tracing skill-agent flow."""
    print("\n" + "=" * 80)
    print(f"DEBUG: {title}")
    if detail:
        print("-" * 80)
        print(detail)
    print("=" * 80)


def debug_log_messages(messages: list, label: str = "messages") -> None:
    """Summarize message list after agent.invoke."""
    lines = [f"{label} ({len(messages)} total):"]
    for i, msg in enumerate(messages):
        msg_type = type(msg).__name__
        content_preview = ""
        if hasattr(msg, "content") and msg.content:
            text = msg.content if isinstance(msg.content, str) else str(msg.content)
            content_preview = text[:120].replace("\n", " ")
            if len(text) > 120:
                content_preview += "..."
        tool_calls = getattr(msg, "tool_calls", None) or []
        tool_info = ""
        if tool_calls:
            names = [tc.get("name", "?") for tc in tool_calls]
            tool_info = f" | tool_calls={names}"
        lines.append(f"  [{i}] {msg_type}{tool_info}: {content_preview}")
    debug_log(f"AGENT RESULT — {label}", "\n".join(lines))

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage

from langchain_groq import ChatGroq

from langchain.agents import create_agent

from langchain.agents.middleware import (
    AgentMiddleware,ModelRequest,ModelResponse
)

from langgraph.checkpoint.memory import InMemorySaver

# LLM
model=ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# skill structure

class Skill(TypedDict):
    name:str
    description:str
    content:str
    
    
# skill DATABASE

SKILLS:list[Skill]=[
    
     {
        "name": "writing_skill",

        "description":
        "Professional writing and grammar assistance.",

        "content":
        """
# Writing Skill

You are an expert writing assistant.

Rules:
- Fix grammar
- Improve clarity
- Make text professional
- Improve sentence structure

Supported:
- Emails
- Articles
- Messages
- Formal writing
"""
    },

    {
        "name": "coding_skill",

        "description":
        "Python programming and debugging help.",

        "content":
        """
# Coding Skill

You are an expert Python developer.

Rules:
- Write clean code
- Add comments
- Explain logic
- Follow best practices

Supported:
- Python
- Debugging
- APIs
- DSA
"""
    }
    
    
]

# Tool -> LOAD SKILL

@tool
def load_skill(skill_name:str) -> str:
    
    """
    load full skill instructions dynamically.
    """
    debug_log("TOOL: load_skill", f"requested skill_name={skill_name!r}")

    for skill in SKILLS:
        if skill["name"] == skill_name:
            debug_log("TOOL: load_skill", f"found skill={skill_name!r}")
            return f""" 
        Loaded skill:{skill_name}
        
        {skill['content']}
        
        """

    available_skills = ",".join(s["name"] for s in SKILLS)
    debug_log(
        "TOOL: load_skill",
        f"skill not found: {skill_name!r}\navailable={available_skills}",
    )
    return f"""
    skills not found.
    Available skills
    {available_skills}
    """
    
    
    # REAL TOOLS
    
    
@tool
def calculator(expression:str)->str:
        """Solve mathematical expressions"""
        debug_log("TOOL: calculator", f"expression={expression!r}")
        result = str(eval(expression))
        debug_log("TOOL: calculator", f"result={result!r}")
        return result
    
@tool
def grammer_fixer(text:str)->str:
        """
        Fix grammer in text
        """
        debug_log("TOOL: grammer_fixer", f"input_len={len(text)} preview={text[:80]!r}")
        corrected=text.capitalize()
        debug_log("TOOL: grammer_fixer", f"output={corrected!r}")
        return corrected
    
@tool
def python_helper(code_topic:str)->str:
        """
        Provide python help
        """
        debug_log("TOOL: python_helper", f"code_topic={code_topic!r}")
        return f"""
    python help for :
    {code_topic}
    """
    
    
    
    # Skills middleware
    
class SkillMiddleware(AgentMiddleware):
    
    # Register tools
    
    tools=[
        load_skill,
        calculator,
        grammer_fixer,
        python_helper
    ]
    
    
    def __init__(self):
        debug_log("SkillMiddleware.__init__", "building skills_prompt from SKILLS registry")
        skills_text=[]
        
        for skill in SKILLS:
            
            skills_text.append(
                f"-{skill['name']}:"
                f"{skill['description']}"
            )
            
            self.skills_prompt="\n".join(skills_text)

        debug_log(
            "SkillMiddleware.__init__ complete",
            f"registered tools={[t.name for t in self.tools]}\n"
            f"skills_prompt:\n{self.skills_prompt}",
        )
        
        
        # Modify system prompt 
        
        
    def wrap_model_call(self, request:ModelRequest, handler:Callable[[ModelRequest],ModelResponse])->ModelResponse:
        debug_log(
            "SkillMiddleware.wrap_model_call",
            "injecting AVAILABLE SKILLS into system message before LLM call",
        )

        # inject avaialble skills
        
        skill_addition=f""" 
        AVAILABLE SKILLS:
        
        {self.skills_prompt}
        
        IMPORTANT:
        
        if user asks something specialized,
        first load the approriate skill
        using load_skill().
        """
        
        
            # Existing system message
        old_content=list(
                request.system_message.content_blocks
            )
    

            # Append skill instrucions
            
        new_content=old_content+[
                {
                    "type":"text",
                    "text":skill_addition
                }
            ]
            
            # create updated system message
            
        new_system_message=SystemMessage(
                content=new_content
            )    
            
            
        updated_request=request.override(
                system_message=new_system_message
            )

        old_block_count = len(old_content)
        new_block_count = len(new_content)
        debug_log(
            "SkillMiddleware.wrap_model_call",
            f"system_message blocks: {old_block_count} -> {new_block_count}\n"
            f"skill_addition preview:\n{skill_addition[:300]}...",
        )

        response = handler(updated_request)
        debug_log("SkillMiddleware.wrap_model_call", "handler returned ModelResponse")
        return response
    
    
    
    

# create agent


debug_log("SETUP", f"creating agent with {len(SKILLS)} skills in registry")

agent=create_agent(
    model=model,
    system_prompt="""
    You are a powerful AI assisstant.
    
    Use skills whenever needed.
    always load skills first before solving specialized tasks
    """,
   
    
    middleware=[
        SkillMiddleware()
    ]
    ,
    checkpointer=InMemorySaver()
    
    )

debug_log("SETUP", "agent created; entering REPL loop (type 'exit' to quit)")

# Run


while True:
    
    user_input=input("\nUSER:")
    
    if user_input.lower()=="exit":
        debug_log("REPL", "user requested exit")
        break

    debug_log("REPL invoke start", f"user_input={user_input!r}\nthread_id=1")

    result=agent.invoke({
        "messages":[
            ("user",user_input)
        ],
        
    },
    config={"configurable": {"thread_id": "1"}} ,                    
    )

    debug_log_messages(result["messages"], label="after invoke")
    debug_log("REPL invoke end", f"final message type={type(result['messages'][-1]).__name__}")
    
    print("\nAI:\n")
    
    print(result["messages"][-1].content)
    
    
    
      
    
    
    