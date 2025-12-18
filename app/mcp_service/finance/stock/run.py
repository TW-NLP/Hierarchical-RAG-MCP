from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from fastmcp import FastMCP

# URL:https://www.modelscope.cn/mcp/servers/@giptilabs/mcp-stock-analysis

stock_app = FastAPI(title="Stock Tools API", version="1.0.0")


# ================== Stock Tools ==================

class StockSymbolInput(BaseModel):
    symbol: str


@stock_app.post("/get_stock_quote", summary="Get the current quote for a stock")
def get_stock_quote(payload: StockSymbolInput):
    """
    获取指定股票的当前报价。
    输入示例:
        {
          "symbol": "RELIANCE.NS"
        }
    输出示例:
        {
          "symbol": "RELIANCE.NS",
          "price": 2748.15,
          "name": "Reliance Industries Ltd"
        }
    """
    return {"success": True, "symbol": payload.symbol, "price": 2748.15, "name": "Reliance Industries Ltd"}


class HistoricalDataInput(BaseModel):
    symbol: str
    interval: Optional[str] = "daily"  # 可选值: "daily", "weekly", "monthly"


@stock_app.post("/get_historical_data", summary="Get historical data for a stock")
def get_historical_data(payload: HistoricalDataInput):
    """
    获取指定股票的历史数据，包含开盘价、最高价、最低价、收盘价、成交量等信息。
    输出结构取决于 interval 参数。
    """
    return {"success": True, "symbol": payload.symbol, "interval": payload.interval, "data": [
        {
            "date": "2023-10-01",
            "open": 2700.00,
            "high": 2750.00,
            "low": 2680.00,
            "close": 2748.15,
            "volume": 1000000
        },
        {
            "date": "2023-10-02",
            "open": 2748.15,
            "high": 2760.00,
            "low": 2720.00,
            "close": 2755.00,
            "volume": 1200000
        }
    ]}  

mcp = FastMCP.from_fastapi(app=stock_app)

if __name__ == '__main__':
    mcp.run(transport='sse',port=50118)