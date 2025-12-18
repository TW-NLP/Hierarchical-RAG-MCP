from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from fastmcp import FastMCP

# URL:https://www.modelscope.cn/mcp/servers/@cantian-ai/Bazi-MCP

bazi_app = FastAPI(title="Bazi Tools API", version="1.0.0")


# ================== 八字详情工具 ==================

class BaziInput(BaseModel):
    solarDatetime: Optional[str] = None   # ISO 格式阳历时间，如 2000-05-15T12:00:00+08:00
    lunarDatetime: Optional[str] = None   # 农历时间，如 2000-05-15 12:00:00
    gender: Optional[int] = 1             # 性别：0 女，1 男，默认 1
    eightCharProviderSect: Optional[int] = 2  # 子时规则：1 为次日，2 为当日，默认 2

@bazi_app.post("/getBaziDetail", summary="Calculate the Bazi results based on the solar/lunar datetime")
def get_bazi_detail(payload: BaziInput):
    return {"success": True,'data':{"bazi": "八字信息"}}

mcp = FastMCP.from_fastapi(app=bazi_app)

if __name__ == '__main__':
    mcp.run(transport='sse',port=50106)