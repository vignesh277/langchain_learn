from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict,Callable

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