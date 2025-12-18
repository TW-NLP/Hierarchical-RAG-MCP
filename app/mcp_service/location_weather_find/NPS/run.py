from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from fastmcp import FastMCP

#URL:https://www.modelscope.cn/mcp/servers/@KyrieTangSheng/mcp-server-nationalparks
parks_app = FastAPI(title="National Parks Tools API", version="1.0.0")


# ========== 1. Find Parks ==========

class FindParksInput(BaseModel):
    stateCode: Optional[str] = None
    q: Optional[str] = None
    limit: Optional[int] = 10
    start: Optional[int] = 0
    activities: Optional[str] = None

@parks_app.post("/findParks", summary="Search for national parks")
def find_parks(payload: FindParksInput):
    return {"success": True}


# ========== 2. Get Park Details ==========

class ParkDetailsInput(BaseModel):
    parkCode: str

@parks_app.post("/getParkDetails", summary="Get detailed information about a national park")
def get_park_details(payload: ParkDetailsInput):
    return {"success": True}


# ========== 3. Get Alerts ==========

class GetAlertsInput(BaseModel):
    parkCode: Optional[str] = None
    limit: Optional[int] = 10
    start: Optional[int] = 0
    q: Optional[str] = None

@parks_app.post("/getAlerts", summary="Get current alerts for parks")
def get_alerts(payload: GetAlertsInput):
    return {"success": True}


# ========== 4. Get Visitor Centers ==========

class GetVisitorCentersInput(BaseModel):
    parkCode: Optional[str] = None
    limit: Optional[int] = 10
    start: Optional[int] = 0
    q: Optional[str] = None

@parks_app.post("/getVisitorCenters", summary="Get visitor center information")
def get_visitor_centers(payload: GetVisitorCentersInput):
    return {"success": True}


# ========== 5. Get Campgrounds ==========

class GetCampgroundsInput(BaseModel):
    parkCode: Optional[str] = None
    limit: Optional[int] = 10
    start: Optional[int] = 0
    q: Optional[str] = None

@parks_app.post("/getCampgrounds", summary="Get campground information")
def get_campgrounds(payload: GetCampgroundsInput):
    return {"success": True}


# ========== 6. Get Events ==========

class GetEventsInput(BaseModel):
    parkCode: Optional[str] = None
    limit: Optional[int] = 10
    start: Optional[int] = 0
    dateStart: Optional[str] = None  # YYYY-MM-DD
    dateEnd: Optional[str] = None
    q: Optional[str] = None

@parks_app.post("/getEvents", summary="Find upcoming events at parks")
def get_events(payload: GetEventsInput):
    return {"success": True}


mcp = FastMCP.from_fastapi(app=parks_app)

if __name__ == '__main__':
    mcp.run(transport='sse',port=50121)