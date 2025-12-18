from fastapi import FastAPI
from pydantic import BaseModel
from fastmcp import FastMCP

#URL:https://www.modelscope.cn/mcp/servers/@modelcontextprotocol/time

timezone_app = FastAPI(title="Timezone MCP Tools API", version="1.0.0")


# ================== Get Current Time ==================

class CurrentTimeInput(BaseModel):
    timezone: str  # 例如: "Asia/Shanghai", "America/New_York"

@timezone_app.post("/get_current_time", summary="Get the current time in a specific time zone.")
def get_current_time(payload: CurrentTimeInput):
    return {"success": True, "data": {"current_time": "2025-08-01T12:00:00+08:00"}}


# ================== Convert Time Between Timezones ==================

class ConvertTimeInput(BaseModel):
    source_timezone: str  # 原始时区（如 "Asia/Shanghai"）
    time: str             # 时间字符串，例如 "13"（24小时制小时数）
    target_timezone: str  # 目标时区（如 "America/New_York"）

@timezone_app.post("/convert_time", summary="Convert time between different time zones")
def convert_time(payload: ConvertTimeInput):
    return {"success": True}

mcp = FastMCP.from_fastapi(app=timezone_app)

if __name__ == '__main__':
    mcp.run(transport='sse',port=50105)