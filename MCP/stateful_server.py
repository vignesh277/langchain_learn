from dotenv import load_dotenv

load_dotenv()

from fastmcp import FastMCP

mcp = FastMCP("StatefulMemory")

_state = {
    "note": None,
    "counter": 0,
}


@mcp.tool()
def set_note(note: str) -> str:
    """Store a note in server memory."""
    _state["note"] = note
    return f"Stored note: {note}"


@mcp.tool()
def get_note() -> str:
    """Get the stored note."""
    note = _state["note"]
    return note if note is not None else "No note stored"


@mcp.tool()
def increment_counter(amount: int = 1) -> int:
    """Increase the in-memory counter."""
    _state["counter"] += amount
    return _state["counter"]


@mcp.tool()
def get_counter() -> int:
    """Get the current counter value."""
    return _state["counter"]


if __name__ == "__main__":
    mcp.run(transport="stdio")
