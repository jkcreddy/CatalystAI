from mcp.server.fastmcp import FastMCP
from langchain_community.tools import DuckDuckGoSearchRun

mcp = FastMCP("hybrid_search")

ddg_search = DuckDuckGoSearchRun()

@mcp.tool()
async def web_search(query: str) -> str:
    """ Search the web using DuckDuckGo if retriever has no results."""
    try:
        result = ddg_search.invoke(query)
        print(f"Web search query: {query}\nWeb search result: {result}")
        return result if result else "No local results found."
    except Exception as e:
        return f"Error performing web search: {str(e)}"
    
if __name__ == "__main__":
    mcp.run(transport="stdio")