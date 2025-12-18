from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from fastmcp import FastMCP

# URL:https://www.modelscope.cn/mcp/servers/@modelcontextprotocol/fetch

fetch_app = FastAPI(title="Fetch Tools API", version="1.0.0")


# ================== Fetch URL Content ==================

class FetchInput(BaseModel):
    url: str                          # 要抓取的 URL，必填
    max_length: Optional[int] = 5000  # 返回最大字符数，默认 5000
    start_index: Optional[int] = 0    # 提取起始字符索引，默认 0
    raw: Optional[bool] = False       # 是否返回原始内容，默认 False（返回 markdown）

@fetch_app.post("/fetch", summary="Fetch a URL and extract its content as markdown.")
def fetch(payload: FetchInput):
    return {"success": True}
mcp=FastMCP.from_fastapi(app=fetch_app)
if __name__ == '__main__':
    mcp.run(transport='sse',port=50101)

