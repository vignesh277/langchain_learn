"""Simple example showing how to use client.get_resources() in MCP."""

import asyncio

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()


async def main() -> None:
    """Load resources from an MCP server and print their contents."""
    client = MultiServerMCPClient(
        {
            "resources": {
                "transport": "http",
                "url": "http://127.0.0.1:8000/mcp",
            }
        }
    )

    blobs = await client.get_resources(
        "resources",
        uris=["file:///path/to/file.txt"],
    )

    for blob in blobs:
        print(f"URI: {blob.metadata['uri']}")
        print(f"MIME type: {blob.mimetype}")
        print(blob.as_string())
        print("---")


if __name__ == "__main__":
    asyncio.run(main())
