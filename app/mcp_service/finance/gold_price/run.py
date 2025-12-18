from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from fastmcp import FastMCP

#URL:https://www.modelscope.cn/mcp/servers/@isdaniel/mcp-metal-price
metal_app = FastAPI(title="Metal Price Tools API", version="1.0.0")


# ================== Metal Price Tool ==================

class MetalPriceInput(BaseModel):
    currency: Optional[str] = "USD"  # ISO 4217 currency code
    metal: Optional[str] = "XAU"     # XAU, XAG, XPT, XPD
    date: Optional[str] = None       # Optional, format: YYYYMMDD

@metal_app.post("/get_gold_price", summary="Get current or historical metal prices")
def get_gold_price(payload: MetalPriceInput):
    """
    获取当前或历史贵金属价格。

    参数说明:
    - currency: 货币代码 (默认: USD)
    - metal: 金属代码 (XAU 金, XAG 银, XPT 铂金, XPD 钯金)
    - date: 可选，历史日期，格式为 YYYYMMDD

    请求示例:
    {
      "currency": "EUR",
      "metal": "XAU"
    }
    """
    return {"success": True}

mcp = FastMCP.from_fastapi(app=metal_app)

if __name__ == '__main__':
    mcp.run(transport='sse',port=50117)