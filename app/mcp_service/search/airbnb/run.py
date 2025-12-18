from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from fastmcp import FastMCP

# URL:https://www.modelscope.cn/mcp/servers/@openbnb-org/mcp-server-airbnb
airbnb_app = FastAPI(title="Airbnb Tools API", version="1.0.0")


# ================== Airbnb Search ==================

class AirbnbSearchInput(BaseModel):
    location: str
    placeId: Optional[str] = None
    checkin: Optional[str] = None  # YYYY-MM-DD
    checkout: Optional[str] = None
    adults: Optional[int] = None
    children: Optional[int] = None
    infants: Optional[int] = None
    pets: Optional[int] = None
    minPrice: Optional[int] = None
    maxPrice: Optional[int] = None
    cursor: Optional[str] = None
    ignoreRobotsText: Optional[bool] = None

@airbnb_app.post("/airbnb_search", summary="Search for Airbnb listings")
def airbnb_search(payload: AirbnbSearchInput):
    return {"success": True}


# ================== Airbnb Listing Details ==================

class AirbnbListingDetailsInput(BaseModel):
    id: str
    checkin: Optional[str] = None
    checkout: Optional[str] = None
    adults: Optional[int] = None
    children: Optional[int] = None
    infants: Optional[int] = None
    pets: Optional[int] = None
    ignoreRobotsText: Optional[bool] = None

@airbnb_app.post("/airbnb_listing_details", summary="Get detailed information about an Airbnb listing")
def airbnb_listing_details(payload: AirbnbListingDetailsInput):
    return {"success": True}


mcp = FastMCP.from_fastapi(app=airbnb_app)

if __name__ == '__main__':
    mcp.run(transport='sse',port=50125)