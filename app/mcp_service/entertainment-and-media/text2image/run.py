from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from fastmcp import FastMCP
#URL=https://www.modelscope.cn/mcp/servers/@zcmmmm/text2image
image_app = FastAPI(title="Text to Image MCP Tools API", version="1.0.0")


class TextToImageInput(BaseModel):
    text: str                                # 必填，待转换文本，可含换行符
    text_color: Optional[str] = "#000000"   # 文字颜色，Hex格式
    bg_color: Optional[str] = "#FFFFFF"     # 背景颜色，Hex格式
    width: Optional[int] = 1080              # 图片宽度，默认1080
    height: Optional[int] = 1080             # 图片高度，默认1080
    font_size: Optional[int] = 80            # 字号，默认80
    font_path: Optional[str] = "simhei.ttf"  # 字体文件路径，默认simhei.ttf
    texture: Optional[str] = None            # 背景纹理图片路径
    output_path: Optional[str] = "output.png" # 输出文件路径，默认output.png
    corner_radius: Optional[int] = 0         # 圆角半径，默认0（无圆角）

@image_app.post("/text_to_image", summary="Convert text to image")
def text_to_image(payload: TextToImageInput):
    # TODO: 这里实现具体的文字转图像逻辑，暂时返回占位
    return {"success": True}


mcp = FastMCP.from_fastapi(app=image_app)

if __name__ == '__main__':
    mcp.run(transport='sse',port=50108)