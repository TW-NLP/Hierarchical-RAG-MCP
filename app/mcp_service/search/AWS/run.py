from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional, List
from fastmcp import FastMCP

app = FastAPI(title="AWS KB Retrieval MCP Service", version="1.0.0")

# ================== 输入输出模型 ==================

class RetrieveFromAwsKbInput(BaseModel):
    query: str = Field(..., description="The search query for retrieval.")
    knowledgeBaseId: str = Field(..., description="The ID of the AWS Knowledge Base.")
    n: Optional[int] = Field(3, description="Number of results to retrieve (default: 3).")

class RetrievalResult(BaseModel):
    id: str
    title: str
    snippet: str

class RetrieveFromAwsKbOutput(BaseModel):
    success: bool
    results: List[RetrievalResult]

# ================== 路由 ==================

@app.post("/retrieve-from-aws-kb", response_model=RetrieveFromAwsKbOutput,summary="Retrieve documents from AWS Knowledge Base")
def retrieve_from_aws_kb(payload: RetrieveFromAwsKbInput):
    """
    Perform retrieval operations using the AWS Knowledge Base.
    """
    # TODO: 实际调用AWS知识库检索接口，以下为示例返回
    example_results = [
        {"id": "doc1", "title": "Sample Document 1", "snippet": "This is a snippet from document 1."},
        {"id": "doc2", "title": "Sample Document 2", "snippet": "This is a snippet from document 2."},
        {"id": "doc3", "title": "Sample Document 3", "snippet": "This is a snippet from document 3."},
    ]
    return {"success": True, "results": example_results[:payload.n]}

# ================== MCP 启动 ==================

mcp = FastMCP.from_fastapi(app=app)

if __name__ == "__main__":
    mcp.run(transport="sse", port=50138)
