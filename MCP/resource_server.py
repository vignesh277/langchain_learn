"""FastMCP server exposing a real text resource."""

from pathlib import Path

from dotenv import load_dotenv
from fastmcp import FastMCP

load_dotenv()

mcp = FastMCP("ResourceDemo")
WELCOME_NOTE_PATH = Path(__file__).with_name("welcome-note.txt")


@mcp.resource("data://welcome-note")
def welcome_note() -> str:
    """Read a real text file and expose it as an MCP resource."""
    return WELCOME_NOTE_PATH.read_text(encoding="utf-8")


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8000)
