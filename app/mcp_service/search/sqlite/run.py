from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from fastmcp import FastMCP

app = FastAPI(title="SQL Tool MCP Server", version="1.0.0")

# ================== 模型定义 ==================

class SQLQueryInput(BaseModel):
    query: str = Field(..., description="SQL query string")

class ReadQueryOutput(BaseModel):
    success: bool
    rows: List[Dict[str, Any]]

class WriteQueryOutput(BaseModel):
    success: bool
    affected_rows: int

class CreateTableOutput(BaseModel):
    success: bool
    message: str

class ListTablesOutput(BaseModel):
    success: bool
    tables: List[str]

class DescribeTableInput(BaseModel):
    table_name: str = Field(..., description="Name of the table to describe")

class DescribeTableOutput(BaseModel):
    success: bool
    columns: List[Dict[str, Any]]

class AppendInsightInput(BaseModel):
    insight: str = Field(..., description="Business insight discovered from data")

class AppendInsightOutput(BaseModel):
    success: bool
    message: str

# ================== 工具定义 ==================

@app.post("/read_query", response_model=ReadQueryOutput,summary="Execute a SELECT query and return result rows")
def read_query(input: SQLQueryInput):
    """Execute SELECT query and return result rows"""
    # TODO: Add actual query execution
    return {"success": True, "rows": []}

@app.post("/write_query", response_model=WriteQueryOutput,summary="Execute an INSERT/UPDATE/DELETE and return affected rows")
def write_query(input: SQLQueryInput):
    """Execute INSERT/UPDATE/DELETE and return affected rows"""
    # TODO: Add actual write execution
    return {"success": True, "affected_rows": 0}

@app.post("/create_table", response_model=CreateTableOutput,summary="Create a new table with SQL")
def create_table(input: SQLQueryInput):
    """Create a new table with SQL"""
    # TODO: Add actual table creation
    return {"success": True, "message": "Table created (mock response)."}

@app.get("/list_tables", response_model=ListTablesOutput,summary="List all tables in the database")
def list_tables():
    """List all tables in the database"""
    # TODO: List actual tables
    return {"success": True, "tables": ["users", "orders", "products"]}

@app.post("/describe-table", response_model=DescribeTableOutput,summary="Describe a table's schema (column names and types)")
def describe_table(input: DescribeTableInput):
    """Describe a table's schema (column names and types)"""
    # TODO: Return actual column info
    return {
        "success": True,
        "columns": [
            {"name": "id", "type": "INTEGER"},
            {"name": "name", "type": "TEXT"}
        ]
    }

@app.post("/append_insight", response_model=AppendInsightOutput,summary="Add new business insight and update memo://insights")
def append_insight(input: AppendInsightInput):
    """Add new business insight and update memo://insights"""
    # TODO: Store insight
    return {"success": True, "message": "Insight added to memo://insights (mock)."}

# ================== MCP 启动 ==================

mcp = FastMCP.from_fastapi(app=app)

if __name__ == '__main__':
    mcp.run(transport='sse', port=50135)
