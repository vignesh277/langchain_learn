"""Simple example showing how to use client.get_resources() in MCP."""

import asyncio
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()


async def main() -> None:
    """Get resources from one or all MCP servers."""
    
    client = MultiServerMCPClient(
        {
            "weather": {
                "transport": "http",
                "url": "http://127.0.0.1:8000/mcp",
            }
        }
    )

    # Example 1: Get all resources from a specific server
    print("=== Example 1: Get all resources from 'weather' server ===")
    resources = await client.get_resources(server_name="weather")
    for resource in resources:
        print(f"Resource: {resource.name}")
        print(f"Mime Type: {resource.mime_type}")
        print(f"Data (first 100 chars): {resource.data[:100]}...")
        print("---")

    # Example 2: Get specific resources by URI
    print("\n=== Example 2: Get specific resources by URI ===")
    resources = await client.get_resources(
        server_name="weather",
        uris="weather://forecast"
    )
    for resource in resources:
        print(f"Resource: {resource.name}")
        print(f"Full data: {resource.data}")

    # Example 3: Get all resources from all servers
    print("\n=== Example 3: Get all resources from all configured servers ===")
    all_resources = await client.get_resources()
    print(f"Total resources retrieved: {len(all_resources)}")
    for resource in all_resources:
        print(f"  - {resource.name} ({resource.mime_type})")


if __name__ == "__main__":
    asyncio.run(main())
