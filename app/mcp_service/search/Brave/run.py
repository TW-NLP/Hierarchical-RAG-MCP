from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from fastmcp import FastMCP

app = FastAPI(title="Search Tools MCP Service", version="1.0.0")

# ========== brave_web_search ==========

class BraveWebSearchInput(BaseModel):
    query: str = Field(..., description="搜索词")
    count: Optional[int] = Field(10, description="每页结果数，最多20")
    offset: Optional[int] = Field(0, description="分页偏移量，最多9")

class BraveWebSearchOutput(BaseModel):
    success: bool
    results: List[Dict[str, Any]] = Field(default_factory=list, description="搜索结果列表")

@app.post("/brave_web_search", response_model=BraveWebSearchOutput,summary="Search the web with pagination and filtering")
def brave_web_search(payload: BraveWebSearchInput):
    """
    执行带分页和筛选的 Web 搜索。
    """
    # TODO: 实现搜索逻辑，当前为模拟返回空结果
    return {"success": True, "results": []}

# ========== brave_local_search ==========

class BraveLocalSearchInput(BaseModel):
    query: str = Field(..., description="本地搜索词")
    count: Optional[int] = Field(10, description="结果数，最多20")

class BraveLocalSearchOutput(BaseModel):
    success: bool
    results: List[Dict[str, Any]] = Field(default_factory=list, description="搜索结果列表")
    fallback_to_web: bool = Field(False, description="如果无本地结果，是否自动回退到 Web 搜索")

@app.post("/brave_local_search", response_model=BraveLocalSearchOutput,summary="Search local businesses and services")
def brave_local_search(payload: BraveLocalSearchInput):
    """
    搜索本地商家和服务，若无结果自动回退到 Web 搜索。
    """
    # TODO: 实现本地搜索及回退逻辑，当前为模拟返回空结果且不回退
    return {"success": True, "results": [], "fallback_to_web": False}


mcp = FastMCP.from_fastapi(app=app)

if __name__ == "__main__":
    mcp.run(transport='sse', port=50139)
