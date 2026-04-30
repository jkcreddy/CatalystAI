import subprocess
import shlex
import os
import urllib.request
import json
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from langchain_community.tools import DuckDuckGoSearchRun

mcp = FastMCP("hybrid_search")

ddg_search = DuckDuckGoSearchRun()

K8S_PROXY = os.getenv("K8S_PROXY_URL", "http://localhost:8001")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCALE_STATE_PATH = PROJECT_ROOT / "logs" / "k8s_scale_state.json"
K8S_NAMESPACES = [
    ns.strip()
    for ns in os.getenv(
        "K8S_NAMESPACES",
        os.getenv("K8S_NAMESPACE", "integration2a,integration3a,integration4a,integration5a,integration6a, loadtest1a,loadtest2a"),
    ).split(",")
    if ns.strip()
]

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


def _pick_active_namespace() -> str:
    """Pick the first configured namespace that currently exists and is Active."""
    try:
        result = subprocess.run(
            [
                "kubectl",
                "get",
                "ns",
                "-o",
                "jsonpath={range .items[*]}{.metadata.name}:{.status.phase}{'\\n'}{end}",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except Exception:
        return K8S_NAMESPACES[0]

    if result.returncode != 0 or not result.stdout.strip():
        return K8S_NAMESPACES[0]

    available: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if ":" not in line:
            continue
        name, phase = line.split(":", 1)
        available[name.strip()] = phase.strip()

    for namespace in K8S_NAMESPACES:
        if available.get(namespace) == "Active":
            return namespace

    return K8S_NAMESPACES[0]


K8S_NAMESPACE = _pick_active_namespace()

NAMESPACED_RESOURCES = {
    "pod", "pods", "deploy", "deployment", "deployments", "service", "services",
    "svc", "replicaset", "replicasets", "rs", "configmap", "configmaps",
    "secret", "secrets", "pvc", "pvcs", "job", "jobs", "cronjob", "cronjobs",
    "daemonset", "daemonsets", "ds", "statefulset", "statefulsets", "sts",
    "ingress", "ingresses", "event", "events",
}

SCALABLE_RESOURCES = ("deployment", "statefulset")


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


def _has_namespace(parts: list[str]) -> bool:
    for i, part in enumerate(parts):
        if part == "-n" and i + 1 < len(parts):
            return True
        if part.startswith("--namespace="):
            return True
    return False


def _inject_namespace(parts: list[str]) -> list[str]:
    """Add the selected active namespace to namespaced kubectl commands when omitted."""
    if len(parts) < 3 or _has_namespace(parts):
        return parts

    subcommand = parts[1]
    resource = parts[2].lower()

    if subcommand in {"get", "describe", "logs", "top", "events"} and resource in NAMESPACED_RESOURCES:
        return [*parts, "-n", K8S_NAMESPACE]

    return parts


def _query_proxy(path: str) -> str:
    url = f"{K8S_PROXY}{path}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode("utf-8")
    except Exception as e:
        return f"Error querying proxy: {str(e)}"


def _load_scale_state() -> dict:
    if not SCALE_STATE_PATH.exists():
        return {}
    try:
        return json.loads(SCALE_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_scale_state(state: dict) -> None:
    SCALE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCALE_STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _kubectl_json(args: list[str]) -> dict:
    result = subprocess.run(
        args + ["-o", "json"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        err = result.stderr.strip() or "kubectl command failed"
        raise RuntimeError(err)
    return json.loads(result.stdout)


def _kubectl_scale(resource_type: str, name: str, replicas: int, namespace: str) -> str:
    result = subprocess.run(
        ["kubectl", "scale", resource_type, name, f"--replicas={replicas}", "-n", namespace],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        err = result.stderr.strip() or "kubectl scale failed"
        raise RuntimeError(err)
    return result.stdout.strip() or f"Scaled {resource_type}/{name} to {replicas}"


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
    parts = _inject_namespace(shlex.split(command.strip()))
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
        executed_command = " ".join(parts)
        return output if output else f"No output returned by command: {executed_command}"
    except subprocess.TimeoutExpired:
        return "Command timed out after 30 seconds"
    except Exception as e:
        return f"Error executing command: {str(e)}"


@mcp.tool()
async def scale_down_workloads(namespace: str | None = None) -> str:
    """Scale deployments/statefulsets down to 1 replica when currently above 1, and save original replica counts."""
    target_ns = (namespace or K8S_NAMESPACE).strip()
    state = _load_scale_state()
    ns_state = state.setdefault(target_ns, {})
    changed: list[str] = []
    skipped: list[str] = []

    try:
        for resource_type in SCALABLE_RESOURCES:
            data = _kubectl_json(["kubectl", "get", resource_type, "-n", target_ns])
            for item in data.get("items", []):
                name = item["metadata"]["name"]
                replicas = int(item.get("spec", {}).get("replicas", 1) or 1)

                if replicas > 1:
                    ns_state[f"{resource_type}/{name}"] = replicas
                    _kubectl_scale(resource_type, name, 1, target_ns)
                    changed.append(f"{resource_type}/{name}: {replicas} -> 1")
                else:
                    skipped.append(f"{resource_type}/{name}: already {replicas}")
    except Exception as e:
        return f"Error scaling down workloads in namespace {target_ns}: {str(e)}"

    _save_scale_state(state)

    lines = [f"Namespace: {target_ns}"]
    if changed:
        lines.append("Scaled down:")
        lines.extend(changed)
    else:
        lines.append("No workloads needed scaling down.")

    if skipped:
        lines.append("")
        lines.append("Skipped:")
        lines.extend(skipped)

    return "\n".join(lines)


@mcp.tool()
async def restore_workloads(namespace: str | None = None) -> str:
    """Restore deployments/statefulsets to the original replica counts saved by scale_down_workloads."""
    target_ns = (namespace or K8S_NAMESPACE).strip()
    state = _load_scale_state()
    ns_state = state.get(target_ns, {})

    if not ns_state:
        return f"No saved scale state found for namespace {target_ns}."

    restored: list[str] = []
    errors: list[str] = []

    for resource_key, replicas in list(ns_state.items()):
        resource_type, name = resource_key.split("/", 1)
        try:
            _kubectl_scale(resource_type, name, int(replicas), target_ns)
            restored.append(f"{resource_key}: restored to {replicas}")
            del ns_state[resource_key]
        except Exception as e:
            errors.append(f"{resource_key}: {str(e)}")

    if ns_state:
        state[target_ns] = ns_state
    else:
        state.pop(target_ns, None)
    _save_scale_state(state)

    lines = [f"Namespace: {target_ns}"]
    if restored:
        lines.append("Restored:")
        lines.extend(restored)
    if errors:
        lines.append("")
        lines.append("Errors:")
        lines.extend(errors)

    return "\n".join(lines)


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
