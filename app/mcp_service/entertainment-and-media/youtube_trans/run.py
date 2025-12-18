from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from fastmcp import FastMCP

#URL=https://www.modelscope.cn/mcp/servers/@kimtaeyoon83/mcp-server-youtube-transcript

youtube_app = FastAPI(title="YouTube Transcript Tools API", version="1.0.0")


# ================== YouTube Transcript Tool ==================

class TranscriptInput(BaseModel):
    url: str                     # 必填：YouTube 视频 URL 或 ID
    lang: Optional[str] = "en"   # 可选：字幕语言代码，默认英文

@youtube_app.post("/get_transcript", summary="Extract subtitles from a YouTube video")
def get_transcript(payload: TranscriptInput):
    return {"success": True}

mcp = FastMCP.from_fastapi(app=youtube_app)

if __name__ == '__main__':
    mcp.run(transport='sse',port=50110)