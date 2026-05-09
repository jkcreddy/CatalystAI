import asyncio
import os

from langchain_mcp_adapters.client import MultiServerMCPClient


def _get_kubernetes_mcp_url() -> str:
    explicit_url = os.getenv("KUBERNETES_MCP_URL")
    if explicit_url:
        return explicit_url

    if os.getenv("KUBERNETES_SERVICE_HOST"):
        return "http://mcp-server-kubernetes-service:3000/sse"

    raise RuntimeError(
        "KUBERNETES_MCP_URL is not set. Configure the remote Kubernetes MCP server URL "
        "or run this client from inside the cluster."
    )


async def main():
    kubernetes_mcp_url = _get_kubernetes_mcp_url()
    async with MultiServerMCPClient({
        "kubernetes": {
            "url": kubernetes_mcp_url,
            "transport": "sse",
        }
    }) as client:

        tools = client.get_tools()
        print("Available tools:", [t.name for t in tools])

        kubectl_tool = next((t for t in tools if t.name == "kubectl_exec"), None)

        if not kubectl_tool:
            raise RuntimeError(f"Missing tools. Available: {[t.name for t in tools]}")

        result = await kubectl_tool.ainvoke({"command": "kubectl get pods"})
        print("\nKubectl Result:\n", result)

if __name__ == "__main__":
    asyncio.run(main())
