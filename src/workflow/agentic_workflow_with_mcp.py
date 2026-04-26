import os
import sys
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

PROJECT_ROOT = "/Users/kjaggav1/Library/CloudStorage/OneDrive-JCPenney/myworkspace/Hackathon/CatalystAI"
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
    def _build_workflow(self):
        workflow = StateGraph(self.AgentState)
        workflow.add_node("Generator", self._generate)
        workflow.add_node("WebSearch", self._web_search)

        workflow.add_edge(START, "WebSearch")
        workflow.add_edge("WebSearch", "Generator")
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
