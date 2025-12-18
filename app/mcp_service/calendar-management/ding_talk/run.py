from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from fastmcp import FastMCP

#URL:https://www.modelscope.cn/mcp/servers/@wllcnm/dingding_mcp_v2
dingtalk_app = FastAPI(title="DingTalk Tools API", version="1.0.0")


# ================== 1. send_message ==================

class SendMessageInput(BaseModel):
    conversation_id: str
    message: str
    msg_type: Optional[str] = "text"

@dingtalk_app.post("/send_message", summary="Send a message to a DingTalk conversation.")
def send_message(payload: SendMessageInput):
    return {"success": True}


# ================== 2. get_conversation_info ==================

class ConversationInfoInput(BaseModel):
    conversation_id: str

@dingtalk_app.post("/get_conversation_info", summary="Get information about a DingTalk conversation.")
def get_conversation_info(payload: ConversationInfoInput):
    return {"success": True}


# ================== 3. get_user_info ==================

class UserInfoInput(BaseModel):
    user_id: str

@dingtalk_app.post("/get_user_info", summary="Get information about a DingTalk user.")
def get_user_info(payload: UserInfoInput):
    return {"success": True}


# ================== 4. get_calendar_list ==================

class CalendarListInput(BaseModel):
    userid: str
    start_time: Optional[int] = None  # 时间戳，毫秒
    end_time: Optional[int] = None
    max_results: Optional[int] = 50
    next_token: Optional[str] = None

@dingtalk_app.post("/get_calendar_list", summary="Get a list of calendar events for a user.")
def get_calendar_list(payload: CalendarListInput):
    return {"success": True}

# Convert FastAPI app to MCP server
mcp = FastMCP.from_fastapi(app=dingtalk_app)

if __name__ == '__main__':
    mcp.run(transport='sse',port=50104)
