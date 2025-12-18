from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from fastmcp import FastMCP

#URL:https://www.modelscope.cn/mcp/servers/@yan5236/bing-cn-mcp-server
bing_app = FastAPI(title="Bing Search Tools API", version="1.0.0")


# ================== Bing Search Tool ==================

class BingSearchInput(BaseModel):
    query: str
    num_results: Optional[int] = 5

@bing_app.post("/bing_search", summary="Search Bing and retrieve the list of results.")
def bing_search(payload: BingSearchInput):
    """
    Use Bing to search for keywords and return a specified number of search results.

    请求示例:
    {
      "query": "OpenAI ChatGPT",
      "num_results": 3
    }
    """
    return {"success": True}


class FetchWebpageInput(BaseModel):
    result_id: str

@bing_app.post("/fetch_webpage", summary="Fetch webpage content by search result ID")
def fetch_webpage(payload: FetchWebpageInput):
    """
    Fetch webpage content by the unique ID of the search result.

    Request example:
    {
      "result_id": "bing-result-123"
    }
    """
    return {"success": True}

mcp = FastMCP.from_fastapi(app=bing_app)

if __name__ == '__main__':
    mcp.run(transport='sse',port=50126)

