from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from fastmcp import FastMCP

#URL:https://www.modelscope.cn/mcp/servers/taopl1990/BiliBiliMCP
bilibili_app = FastAPI(title="Bilibili Video Tools API", version="1.0.0")


# 1. load_bilibili_video
class LoadBilibiliVideoInput(BaseModel):
    video_url: str
    use_auth: Optional[bool] = True

@bilibili_app.post("/load_bilibili_video", summary="Load Bilibili video content into memory.")
def load_bilibili_video(payload: LoadBilibiliVideoInput):
    return {"success": True}


# 2. get_video_content
class GetVideoContentInput(BaseModel):
    video_id: str
    page_index: Optional[int] = None

@bilibili_app.post("/get_video_content", summary="Get content of the loaded video")
def get_video_content(payload: GetVideoContentInput):
    return {"success": True}


# 3. search_in_video
class SearchInVideoInput(BaseModel):
    video_id: str
    query: str
    case_sensitive: Optional[bool] = False

@bilibili_app.post("/search_in_video", summary="Search for specific text in the video content")
def search_in_video(payload: SearchInVideoInput):
    return {"success": True}


# 4. extract_video_summary
class ExtractVideoSummaryInput(BaseModel):
    video_id: str
    max_length: Optional[int] = 500

@bilibili_app.post("/extract_video_summary", summary="Extract video content summary")
def extract_video_summary(payload: ExtractVideoSummaryInput):
    return {"success": True}


# 5. export_video_content
class ExportVideoContentInput(BaseModel):
    video_id: str
    target_format: Optional[str] = "txt"  # txt/json/srt
    output_path: Optional[str] = None

@bilibili_app.post("/export_video_content", summary="Export video content to file")
def export_video_content(payload: ExportVideoContentInput):
    return {"success": True}


# 6. get_bilibili_auth_status
@bilibili_app.get("/get_bilibili_auth_status", summary="Check Bilibili authentication status")
def get_bilibili_auth_status():
    return {"success": True}

mcp = FastMCP.from_fastapi(app=bilibili_app)

if __name__ == '__main__':
    mcp.run(transport='sse',port=50107)
