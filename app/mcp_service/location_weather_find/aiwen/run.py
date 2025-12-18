from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from fastmcp import FastMCP

app = FastAPI(title="IP Location MCP Service", version="1.0.0")


# ========== Input / Output Models ==========

class IPQueryInput(BaseModel):
    ip: str = Field(..., description="IPv4 address to query")


class IPLocationOutput(BaseModel):
    code: str
    data: Dict[str, Any]
    charge: bool
    msg: str
    ip: str
    coordsys: Optional[str] = "WGS84"


# ========== Tool 1: aiwen_ip_location ==========

@app.post("/aiwen_ip_location", response_model=IPLocationOutput,summary="Get detailed information about an IP address")
def aiwen_ip_location(payload: IPQueryInput):
    """Returns detailed information about the requested IP address"""
    ip = payload.ip

    # TODO: 实际上你应通过外部API调用，例如 ip-api.com, ipinfo.io 或你自有接口
    # 以下为模拟数据
    example_data = {
        "code": "Success",
        "data": {
            "continent": "Asia",
            "country": "China",
            "owner": "China Telecom",
            "isp": "China Telecom",
            "zipcode": "510000",
            "timezone": "UTC+8",
            "accuracy": "City",
            "source": "Data Mining",
            "areacode": "CN",
            "adcode": "440100",
            "asnumber": "4134",
            "lat": "23.116548",
            "lng": "113.295827",
            "radius": "87.3469",
            "prov": "Guangdong Province",
            "city": "Guangzhou"
        },
        "charge": True,
        "msg": "Query successful",
        "ip": ip,
        "coordsys": "WGS84"
    }

    return example_data


# ========== Tool 2: user_network_ip ==========

@app.get("/user_network_ip", response_model=IPLocationOutput,summary="Get current network IP address and location information")
def user_network_ip():
    """Returns current network IP address and location information"""
    # TODO: 实际应用中应获取真实 IP，例如通过 request.client.host 或反向代理头

    fake_ip = "202.97.89.109"
    # 复用上面的模拟响应
    return aiwen_ip_location(IPQueryInput(ip=fake_ip))


# ========== MCP 启动 ==========

mcp = FastMCP.from_fastapi(app=app)

if __name__ == '__main__':
    mcp.run(transport="sse", port=50133)
