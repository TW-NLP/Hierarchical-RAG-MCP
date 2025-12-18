from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from fastmcp import FastMCP

# URL:https://www.modelscope.cn/mcp/servers/@arjunprabhulal/mcp-flight-search
flight_app = FastAPI(title="Flight MCP Tools API", version="1.0.0")


# ================== Flight Search Tool ==================

class FlightSearchInput(BaseModel):
    origin: str                     # 出发机场代码（例如 ATL, JFK）
    destination: str                # 到达机场代码（例如 LAX, ORD）
    outbound_date: str              # 出发日期，格式 YYYY-MM-DD
    return_date: Optional[str] = None  # 返回日期（可选）

@flight_app.post("/search_flights_tool", summary="Search for flights between airports")
def search_flights_tool(payload: FlightSearchInput):
    return {"success": True}


# ================== Server Status Check ==================

@flight_app.get("/server_status", summary="Check if the MCP server is running")
def server_status():
    return {"status": "running", "success": True}


mcp = FastMCP.from_fastapi(app=flight_app)

if __name__ == '__main__':
    mcp.run(transport='sse',port=50127)
