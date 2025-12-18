from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from fastmcp import FastMCP

app = FastAPI(title="Google Maps MCP Server", version="1.0.0")

# ===================== 模型定义 =====================

class GeocodeInput(BaseModel):
    address: str = Field(..., description="The address to geocode")

class GeocodeOutput(BaseModel):
    location: Dict[str, float]  # {latitude, longitude}
    formatted_address: str
    place_id: str

class ReverseGeocodeInput(BaseModel):
    latitude: float
    longitude: float

class ReverseGeocodeOutput(BaseModel):
    formatted_address: str
    place_id: str
    address_components: List[Dict[str, Any]]

class SearchPlacesInput(BaseModel):
    query: str
    location: Optional[Dict[str, float]] = Field(None, description="Optional location bias")
    radius: Optional[int] = Field(5000, description="Search radius in meters (max 50000)")

class PlaceSummary(BaseModel):
    name: str
    address: str
    location: Dict[str, float]

class SearchPlacesOutput(BaseModel):
    places: List[PlaceSummary]

class PlaceDetailsInput(BaseModel):
    place_id: str

class PlaceDetailsOutput(BaseModel):
    name: str
    address: str
    contact_info: Optional[str]
    ratings: Optional[float]
    reviews: Optional[List[str]]
    opening_hours: Optional[Dict[str, Any]]

class DistanceMatrixInput(BaseModel):
    origins: List[str]
    destinations: List[str]
    mode: Optional[str] = Field("driving", description="Mode of transport")

class DistanceMatrixOutput(BaseModel):
    distances: List[List[str]]
    durations: List[List[str]]

class ElevationInput(BaseModel):
    locations: List[Dict[str, float]]  # [{"latitude": ..., "longitude": ...}]

class ElevationOutput(BaseModel):
    elevations: List[float]

class DirectionsInput(BaseModel):
    origin: str
    destination: str
    mode: Optional[str] = Field("driving", description="driving, walking, bicycling, transit")

class DirectionsOutput(BaseModel):
    steps: List[str]
    total_distance: str
    total_duration: str


# ===================== 工具路由实现 =====================

@app.post("/maps_geocode", response_model=GeocodeOutput,summary="Use the Google Maps Geocoding API to obtain geocoding information.")
def maps_geocode(payload: GeocodeInput):
    # TODO: 使用 Google Maps Geocoding API 获取数据
    return {
        "location": {"latitude": 35.6895, "longitude": 139.6917},
        "formatted_address": "Tokyo, Japan",
        "place_id": "abcdef123456"
    }

@app.post("/maps_reverse_geocode", response_model=ReverseGeocodeOutput,summary="Use the Google Maps Reverse Geocoding API to obtain reverse geocoding information.")
def maps_reverse_geocode(payload: ReverseGeocodeInput):
    # TODO: 使用 Google Maps Reverse Geocoding API
    return {
        "formatted_address": "Chiyoda City, Tokyo, Japan",
        "place_id": "xyz987654321",
        "address_components": [{"long_name": "Tokyo", "types": ["locality"]}]
    }

@app.post("/maps_search_places", response_model=SearchPlacesOutput,summary="Use the Google Maps Places Text Search API to search for places.")
def maps_search_places(payload: SearchPlacesInput):
    # TODO: 使用 Google Maps Places Text Search API
    return {
        "places": [
            {"name": "Tokyo Tower", "address": "4 Chome-2-8 Shibakoen, Minato City", "location": {"latitude": 35.6586, "longitude": 139.7454}},
            {"name": "Tokyo Skytree", "address": "1 Chome-1-2 Oshiage, Sumida City", "location": {"latitude": 35.7101, "longitude": 139.8107}}
        ]
    }

@app.post("/maps_place_details", response_model=PlaceDetailsOutput,summary="Use the Google Maps Place Details API to obtain details about a place.")
def maps_place_details(payload: PlaceDetailsInput):
    # TODO: 使用 Google Maps Place Details API
    return {
        "name": "Tokyo Tower",
        "address": "4 Chome-2-8 Shibakoen, Minato City",
        "contact_info": "+81 3-3433-5111",
        "ratings": 4.6,
        "reviews": ["Great view!", "Amazing experience!"],
        "opening_hours": {"monday": "9:00–23:00"}
    }

@app.post("/maps_distance_matrix", response_model=DistanceMatrixOutput,summary="Use the Google Maps Distance Matrix API to obtain distance matrix")
def maps_distance_matrix(payload: DistanceMatrixInput):
    # TODO: 使用 Google Maps Distance Matrix API
    return {
        "distances": [["5.6 km", "8.2 km"]],
        "durations": [["12 mins", "20 mins"]]
    }

@app.post("/maps_elevation", response_model=ElevationOutput,summary="Use the Google Maps Elevation API to obtain elevation data")
def maps_elevation(payload: ElevationInput):
    # TODO: 使用 Google Maps Elevation API
    return {
        "elevations": [44.5, 35.7]
    }

@app.post("/maps_directions", response_model=DirectionsOutput,summary="Use the Google Maps Directions API to obtain route planning")
def maps_directions(payload: DirectionsInput):
    # TODO: 使用 Google Maps Directions API
    return {
        "steps": ["Head north", "Turn left at traffic light", "Arrive at destination"],
        "total_distance": "6.4 km",
        "total_duration": "15 mins"
    }

# ===================== 启动服务 =====================

mcp = FastMCP.from_fastapi(app=app)

if __name__ == '__main__':
    mcp.run(transport='sse', port=50134)
