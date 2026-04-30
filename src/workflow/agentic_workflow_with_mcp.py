import os
import sys
from pathlib import Path
from typing import Annotated, Sequence, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from prompt_library.prompts import PROMPT_REGISTRY, PromptType
#from retriever.retrieval import Retriever
from utils.model_loader import ModelLoader
#from evaluation.ragas_eval import evaluate_context_precision, evaluate_response_relevancy
from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
SERVER_PATH = f"{PROJECT_ROOT}/src/mcp_servers/server.py"


class AgenticAI:
    """Agentic AI pipeline using LangGraph + MCP (Retriever + Web Search)"""

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]

    def __init__(self):
        #self.retriever_obj = Retriever()
        self.model_loader = ModelLoader()
        self.llm = self.model_loader.load_llm()
        self.checkpointer = MemorySaver()

        # MCP tools are initialized per run inside an active event loop.
        self.mcp_tools = []

        self.workflow = self._build_workflow()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)

    # ----------Helper Methods----------
    def _build_mcp_client(self) -> MultiServerMCPClient:
        return MultiServerMCPClient({
            "product_retriever": {
                "command": sys.executable,
                "args": [SERVER_PATH],
                "transport": "stdio",
                "env": {
                    **os.environ,
                    "PYTHONPATH": PROJECT_ROOT
                }
            }
        })
    
    # ---------- Nodes ----------
    @staticmethod
    def _normalize_kubectl_command(raw_command: str) -> str:
        """Strip markdown/code-fence formatting and keep only the kubectl command."""
        command = raw_command.strip()

        if command.startswith("```"):
            lines = [line.rstrip() for line in command.splitlines()]
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            command = "\n".join(lines).strip()

        if command.lower().startswith("bash"):
            command = command[4:].strip()

        if command.lower().startswith("kubectl command:"):
            command = command.split(":", 1)[1].strip()

        for line in command.splitlines():
            stripped = line.strip()
            if stripped.startswith("kubectl "):
                return stripped

        return command

    @staticmethod
    def _is_restore_request(query: str) -> bool:
        lowered = query.lower()
        return (
            "scale up" in lowered
            or "restore" in lowered
            or "original scale" in lowered
            or "back to original" in lowered
            or "previous scale" in lowered
        )

    @staticmethod
    def _is_k8s_issue_query(query: str) -> bool:
        lowered = query.lower()
        issue_keywords = [
            "down", "not working", "not running", "failing", "failed", "crash",
            "crashloop", "error", "unhealthy", "pending", "stuck", "restart",
            "restarts", "unavailable", "503", "500", "timeout", "why is", "reason",
        ]
        return any(keyword in lowered for keyword in issue_keywords)

    def _normalize_command_list(self, raw_commands: str) -> list[str]:
        commands: list[str] = []
        for line in raw_commands.splitlines():
            normalized = self._normalize_kubectl_command(line)
            if normalized.startswith("kubectl ") and normalized not in commands:
                commands.append(normalized)
        return commands[:3]

    def _ai_assistant(self, state: AgentState):
        print("--- CALL AI ASSISTANT ---")
        messages = state["messages"]
        last_message = messages[-1].content

        # if any(word in last_message.lower() for word in ["price", "review", "product"]):
        return {"messages": [HumanMessage(content=f"TOOL: web_search\nQUERY: {last_message}")]}
        # else:
        #     prompt = ChatPromptTemplate.from_template(
        #         "You are a helpful assistant. Answer the user directly.\n\nQuestion: {question}\n\nAnswer:"
        #     )
        #     chain = prompt | self.llm | StrOutputParser()
        #     response = chain.invoke({"question": last_message}) or "I'm not sure about that."
        #     return {"messages": [HumanMessage(content=response)]}
        
    # async def _vector_retriever(self, state: AgentState):
    #     print("--- RETRIEVER (MCP) ---")
    #     last_message = state["messages"][-1].content
    #     query = last_message.split("QUERY:", 1)[1].strip() if "QUERY:" in last_message else state["messages"][0].content

    #     tool = next((t for t in self.mcp_tools if t.name == "get_product_info"), None)
    #     if not tool:
    #         return {"messages": [HumanMessage(content="No retriever tool available in MCP client.")]}
        
    #     try:
    #         result = await tool.ainvoke({"query": query})
    #         context = result or "No relevant documents found."
    #     except Exception as e:
    #         context = f"Error calling retriever tool: {str(e)}"

    #     return {"messages": [HumanMessage(content=context)]}
    
    async def _web_search(self, state: AgentState):
        print("---- WEB SEARCH (MCP) ---")
        query = state["messages"][0].content.strip()
        if query.lower().startswith("rewritten query:"):
            query = query.split(":", 1)[1].strip()

        tool = next((t for t in self.mcp_tools if t.name == "web_search"), None)
        if not tool:
            return {"messages": [HumanMessage(content="No web_search tool available in MCP client.")]}

        result = await tool.ainvoke({"query": query})
        context = result if result else "No data from web"
        return {"messages": [HumanMessage(content=context)]}

    async def _kubectl(self, state: AgentState):
        print("---- KUBECTL (MCP) ---")
        query = state["messages"][0].content.strip()

        tool = next((t for t in self.mcp_tools if t.name == "kubectl_exec"), None)
        if not tool:
            return {"messages": [HumanMessage(content="No kubectl_exec tool available in MCP client.")]}

        if self._is_k8s_issue_query(query):
            prompt = ChatPromptTemplate.from_template(
                "The user is asking for the reason a Kubernetes resource is down or unhealthy. "
                "Generate up to 3 plain-text read-only kubectl commands, one per line, that help diagnose the issue. "
                "Prefer commands like describe, logs, get pods, get events, and get endpoints. "
                "Do not use markdown, bullets, numbering, code fences, explanations, or write commands. "
                "Every line must start with kubectl.\n\n"
                "User request: {question}"
            )
            chain = prompt | self.llm | StrOutputParser()
            raw_commands = chain.invoke({"question": query}).strip()
            kubectl_commands = self._normalize_command_list(raw_commands)

            if not kubectl_commands:
                kubectl_commands = ["kubectl get pods"]

            sections = [f"Kubernetes diagnosis for: {query}"]
            for index, kubectl_cmd in enumerate(kubectl_commands, start=1):
                result = await tool.ainvoke({"command": kubectl_cmd})
                sections.append(f"\nCommand {index}: {kubectl_cmd}")
                sections.append(f"Result {index}:\n{result}")

            return {"messages": [HumanMessage(content="\n".join(sections))]}

        prompt = ChatPromptTemplate.from_template(
            "Convert the following user request into a valid kubectl command. "
            "Return only a single plain-text kubectl command. "
            "Do not wrap the command in markdown, code fences, backticks, or any explanation. "
            "Do not prefix the answer with 'bash' or 'kubectl command:'. "
            "Only use read-only commands: get, describe, logs, top, explain, events, cluster-info, version. "
            "Your response must start with the word kubectl.\n\n"
            "User request: {question}\n\nkubectl"
        )
        chain = prompt | self.llm | StrOutputParser()
        kubectl_cmd = self._normalize_kubectl_command(
            chain.invoke({"question": query}).strip()
        )
        result = await tool.ainvoke({"command": kubectl_cmd})
        context = f"Command: {kubectl_cmd}\n\nResult:\n{result}"
        return {"messages": [HumanMessage(content=context)]}

    async def _scale_workloads(self, state: AgentState):
        print("---- SCALE WORKLOADS (MCP) ---")
        query = state["messages"][0].content.strip()
        tool_name = "restore_workloads" if self._is_restore_request(query) else "scale_down_workloads"
        tool = next((t for t in self.mcp_tools if t.name == tool_name), None)

        if not tool:
            return {"messages": [HumanMessage(content=f"No {tool_name} tool available in MCP client.")]}

        result = await tool.ainvoke({})
        action = "Restore original scale" if tool_name == "restore_workloads" else "Scale down workloads"
        context = f"Action: {action}\n\nResult:\n{result}"
        return {"messages": [HumanMessage(content=context)]}
    
    # def _grade_documents(self, state: AgentState) -> Literal["generator", "rewriter"]:
    #     print("--- GRADER ---")
    #     question = state["messages"][0].content
    #     docs = state["messages"][1].content

    #     prompt = PromptTemplate(
    #         template="""You are a grader.Question: {question}\nDocs: {docs}\n
    #         Are docs relevant to the question? Answer yes or no.""",
    #         input_variables=["question", "docs"]
    #     )
    #     chain = prompt | self.llm | StrOutputParser()
    #     score = chain.invoke({"question": question, "docs": docs}) or ""
    #     return "generator" if "yes" in score.lower() else "rewriter"
    
    def _generate(self, state: AgentState):
        print("--- GENERATOR ---")
        question = state["messages"][0].content
        docs = state["messages"][-1].content

        prompt = ChatPromptTemplate.from_template(
            PROMPT_REGISTRY[PromptType.KUBERNETES_BOT].template
        )
        chain = prompt | self.llm | StrOutputParser()

        try:
            response = chain.invoke({"question": question, "context": docs}) or "I'm not sure based on the information I have."
        except Exception as e:
            response = f"Error generating response: {str(e)}"

        return { "messages": [HumanMessage(content=response)] }
    
    def _rewrite(self, state: AgentState):
        print("--- REWRITE ---")
        question = state["messages"][0].content

        prompt = ChatPromptTemplate.from_template(
            "Rewrite this user query to make it more clear and specific for a search engine."
            "Do NOT answer the query. Only rewrite it.\n\nQuery: {question}\nRewritten Query:"
        )
        chain = prompt | self.llm | StrOutputParser()

        try:
            new_q = chain.invoke({"question": question}).strip()
        except Exception as e:
            new_q = f"Error rewriting query: {str(e)}"

        if new_q.lower().startswith("rewritten query:"):
            new_q = new_q.split(":", 1)[1].strip()

        return {"messages": [HumanMessage(content=new_q)]}
    
    # ---------- Workflow ----------
    def _route_after_kubectl(self, state: AgentState) -> Literal["Generator", "__end__"]:
        query = state["messages"][0].content
        if self._is_k8s_issue_query(query):
            return "Generator"
        return END

    def _route_query(self, state: AgentState) -> Literal["Kubectl", "ScaleWorkloads", "WebSearch"]:
        query = state["messages"][0].content.lower()
        scaling_keywords = ["scale down", "scale up", "restore", "original scale", "replicas", "replica count"]
        k8s_keywords = ["pod", "pods", "deploy", "deployment", "service", "node", "namespace",
                        "kubectl", "logs", "describe", "cluster", "replica", "ingress",
                        "configmap", "secret", "pvc", "pv", "daemonset", "statefulset",
                        "cronjob", "job", "hpa", "events", "container", "k8s", "kubernetes"]
        if any(kw in query for kw in scaling_keywords):
            return "ScaleWorkloads"
        if any(kw in query for kw in k8s_keywords):
            return "Kubectl"
        return "WebSearch"

    def _build_workflow(self):
        workflow = StateGraph(self.AgentState)
        workflow.add_node("Generator", self._generate)
        workflow.add_node("WebSearch", self._web_search)
        workflow.add_node("Kubectl", self._kubectl)
        workflow.add_node("ScaleWorkloads", self._scale_workloads)

        workflow.add_conditional_edges(START, self._route_query)
        workflow.add_edge("WebSearch", "Generator")
        workflow.add_conditional_edges("Kubectl", self._route_after_kubectl)
        workflow.add_edge("ScaleWorkloads", END)
        workflow.add_edge("Generator", END)

        return workflow
    
    # ---------- Public Run ----------
    async def arun(self, query: str, thread_id: str = "default_thread") -> str:
        """Run the agentic RAG workflow with the given user query."""
        async with self._build_mcp_client() as client:
            self.mcp_tools = client.get_tools()
            result = await self.app.ainvoke(
                {"messages": [HumanMessage(content=query)]},
                config={"configurable": {"thread_id": thread_id}},
            )
        return result["messages"][-1].content

    def run(self, query: str, thread_id: str = "default_thread") -> str:
        """Synchronous wrapper."""
        return asyncio.run(self.arun(query, thread_id=thread_id))
    
if __name__ == "__main__":
    agentic_ai = AgenticAI()
    user_query = "What is the capital of India?"
    response = agentic_ai.run(query=user_query)
    print("Final Response:", response)
