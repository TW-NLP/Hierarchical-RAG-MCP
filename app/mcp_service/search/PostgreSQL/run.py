from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from fastmcp import FastMCP

# 初始化 FastAPI 应用
app = FastAPI(title="PostgreSQL Read-Only MCP Server", version="1.0.0")

# ====================== 输入输出模型定义 ======================

class QueryInput(BaseModel):
    sql: str = Field(..., description="The SQL query to execute (must be SELECT/read-only)")

class QueryOutput(BaseModel):
    success: bool
    rows: List[Dict[str, Any]] = Field(..., description="The result rows from the database")

# ====================== 工具接口定义 ======================

@app.post("/query", response_model=QueryOutput,summary="Execute a read-only SQL query against the PostgreSQL database")
def query_database(payload: QueryInput):
    """
    Execute a read-only SQL query against the connected PostgreSQL database.

    Input:
    - sql (string): The SQL query to execute. Only SELECT queries are allowed.
    
    Output:
    - rows (list of dicts): The query results (each row as a dictionary)
    """
    # ⚠️ 模拟返回，不实际执行查询
    if not payload.sql.strip().lower().startswith("select"):
        raise HTTPException(status_code=400, detail="Only SELECT queries are allowed.")

    fake_rows = [
        {"id": 1, "name": "Example User"},
        {"id": 2, "name": "Another User"}
    ]
    return {"success": True, "rows": fake_rows}

# ====================== 启动 MCP 服务 ======================

mcp = FastMCP.from_fastapi(app=app)

if __name__ == "__main__":
    mcp.run(transport="sse", port=50136)
