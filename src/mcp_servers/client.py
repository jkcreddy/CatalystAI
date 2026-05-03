import asyncio
import os
import sys

from langchain_mcp_adapters.client import MultiServerMCPClient

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER_PATH = f"{PROJECT_ROOT}/mcp_servers/server.py"
KUBERNETES_MCP_URL = os.getenv(
    "KUBERNETES_MCP_URL",
    "http://mcp-server-kubernetes-service:8080/sse",
)

async def main():
    async with MultiServerMCPClient({
        "hybrid_search": {
            "command": sys.executable,
            "args": [SERVER_PATH],
            "transport": "stdio",
            "env": {
                **os.environ,
                "PYTHONPATH": PROJECT_ROOT
            }
        },
        "kubernetes": {
            "url": KUBERNETES_MCP_URL,
            "transport": "sse",
        }
    }) as client:

        tools = client.get_tools()
        print("Available tools:", [t.name for t in tools])

        web_search_tool = next((t for t in tools if t.name == "web_search"), None)

        if not web_search_tool:
            raise RuntimeError(f"Missing tools. Available: {[t.name for t in tools]}")

        query = "Hi How are you?"
        web_search_result = await web_search_tool.ainvoke({"query": query})
        print("\nWeb Search Result:\n", web_search_result)

        if (not str(web_search_result).strip()) or ("No local results found." in str(web_search_result)):
            print("\nWeb Search failed.")

if __name__ == "__main__":
    asyncio.run(main())
