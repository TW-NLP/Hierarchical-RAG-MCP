from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
from fastmcp import FastMCP

app = FastAPI(title="Google Drive MCP Server", version="1.0.0")

# ================ Tool: search ================

class SearchInput(BaseModel):
    query: str = Field(..., description="Search query string for Google Drive files")

class SearchResultItem(BaseModel):
    file_name: str
    mime_type: str

class SearchOutput(BaseModel):
    success: bool
    results: List[SearchResultItem]

@app.post("/search", response_model=SearchOutput,summary="Search for files in Google Drive by query string")
def search_files(payload: SearchInput):
    """
    Search for files in Google Drive by query string.
    """
    # TODO: 集成 Google Drive API 实现搜索功能
    example_results = [
        {"file_name": "Report.pdf", "mime_type": "application/pdf"},
        {"file_name": "Presentation.pptx", "mime_type": "application/vnd.ms-powerpoint"}
    ]
    return {"success": True, "results": example_results}


# ================ MCP 启动 ================

mcp = FastMCP.from_fastapi(app=app)

if __name__ == '__main__':
    mcp.run(transport='sse', port=50140)
