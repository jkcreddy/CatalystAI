import subprocess
import shlex
import os
import urllib.request
import json
from mcp.server.fastmcp import FastMCP
from langchain_community.tools import DuckDuckGoSearchRun

mcp = FastMCP("hybrid_search")

ddg_search = DuckDuckGoSearchRun()

K8S_PROXY = os.getenv("K8S_PROXY_URL", "http://localhost:8001")
K8S_NAMESPACE = os.getenv("K8S_NAMESPACE", "integration4a")

ALLOWED_KUBECTL_COMMANDS = {
    "get", "describe", "logs", "top", "explain",
    "api-resources", "api-versions", "cluster-info",
    "config", "version", "events",
}

BLOCKED_KUBECTL_COMMANDS = {
    "delete", "apply", "create", "patch", "edit",
    "replace", "scale", "rollout", "exec", "run",
    "cordon", "uncordon", "drain", "taint", "label",
    "annotate", "expose",
}


def _validate_kubectl(cmd: str) -> str | None:
    parts = shlex.split(cmd.strip())
    if not parts or parts[0] != "kubectl":
        return "Command must start with 'kubectl'"
    if len(parts) < 2:
        return "Incomplete kubectl command"
    subcommand = parts[1]
    if subcommand in BLOCKED_KUBECTL_COMMANDS:
        return f"'{subcommand}' is blocked — only read-only commands are allowed"
    if subcommand not in ALLOWED_KUBECTL_COMMANDS:
        return f"'{subcommand}' is not in the allowed commands list"
    return None


def _query_proxy(path: str) -> str:
    url = f"{K8S_PROXY}{path}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode("utf-8")
    except Exception as e:
        return f"Error querying proxy: {str(e)}"


@mcp.tool()
async def kubectl_exec(command: str) -> str:
    """Execute a read-only kubectl command against the Kubernetes cluster.
    Allowed: get, describe, logs, top, explain, api-resources, api-versions, cluster-info, config, version, events.
    Blocked: delete, apply, create, patch, edit, exec, scale, and other write operations.
    Example: kubectl get pods -n integration4a
    """
    error = _validate_kubectl(command)
    if error:
        return f"Rejected: {error}"

    # Try kubectl proxy first, fall back to direct kubectl
    parts = shlex.split(command.strip())
    subcommand = parts[1]

    # Attempt via kubectl proxy for common get commands
    if subcommand == "get" and K8S_PROXY:
        ns = K8S_NAMESPACE
        resource = parts[2] if len(parts) > 2 else "pods"

        # Extract namespace from command if specified
        for i, p in enumerate(parts):
            if p == "-n" and i + 1 < len(parts):
                ns = parts[i + 1]
                break
            if p.startswith("--namespace="):
                ns = p.split("=", 1)[1]
                break

        api_path = f"/api/v1/namespaces/{ns}/{resource}"
        raw = _query_proxy(api_path)

        if not raw.startswith("Error"):
            try:
                data = json.loads(raw)
                items = data.get("items", [])
                if not items:
                    return f"No {resource} found in namespace {ns}"

                lines = [f"{'NAME':<60} {'STATUS':<20} {'RESTARTS':<10} {'AGE'}"]
                for item in items:
                    name = item["metadata"]["name"]
                    phase = item.get("status", {}).get("phase", "Unknown")
                    container_statuses = item.get("status", {}).get("containerStatuses", [])
                    restarts = sum(cs.get("restartCount", 0) for cs in container_statuses)

                    # Calculate age
                    start = item["metadata"].get("creationTimestamp", "")
                    lines.append(f"{name:<60} {phase:<20} {restarts:<10} {start}")

                return "\n".join(lines)
            except json.JSONDecodeError:
                pass

    # Strip streaming flags and add tail limit for logs
    if subcommand == "logs":
        parts = [p for p in parts if p not in ("-f", "--follow", "-w", "--watch")]
        if "--tail" not in command:
            parts.append("--tail=200")

    # Fallback to direct kubectl
    try:
        result = subprocess.run(
            parts,
            capture_output=True, text=True, timeout=30,
        )
        output = result.stdout.strip() if result.stdout else ""
        err = result.stderr.strip() if result.stderr else ""
        if result.returncode != 0:
            return f"Error:\n{err}" if err else "Command failed with no output"
        return output if output else "Command executed successfully (no output)"
    except subprocess.TimeoutExpired:
        return "Command timed out after 30 seconds"
    except Exception as e:
        return f"Error executing command: {str(e)}"


@mcp.tool()
async def web_search(query: str) -> str:
    """Search the web using DuckDuckGo if retriever has no results."""
    try:
        result = ddg_search.invoke(query)
        print(f"Web search query: {query}\nWeb search result: {result}")
        return result if result else "No local results found."
    except Exception as e:
        return f"Error performing web search: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
