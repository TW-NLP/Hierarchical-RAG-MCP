from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import uuid
import os

from fastmcp import FastMCP
#URL:https://modelscope.cn/mcp/servers/@rmcendarfer2017/MCP-image-gen
app = FastAPI(title="Image Tools API", version="1.0.0")

# Directory to save images
IMAGE_DIR = "generated_images"
os.makedirs(IMAGE_DIR, exist_ok=True)


# ================== Generate Image Tool ==================
class GenerateImageInput(BaseModel):
    prompt: str = Field(..., description="Text prompt for image generation")
    negative_prompt: str = Field(None, description="Text to avoid in generated image")
    width: int = Field(512, description="Width of generated image")
    height: int = Field(512, description="Height of generated image")
    num_inference_steps: int = Field(50, description="Number of diffusion steps")
    guidance_scale: float = Field(7.5, description="Classifier-free guidance scale")

class GenerateImageOutput(BaseModel):
    success: bool
    image_url: str
    metadata: Dict[str, Any]

@app.post("/generate-image", response_model=GenerateImageOutput, summary="Generate an image using Stable Diffusion")
def generate_image(payload: GenerateImageInput):
    """
    Generate an image based on the provided prompt and return its URL and metadata.
    """
    # TODO: Integrate with Replicate's Stable Diffusion API
    # Placeholder implementation
    fake_url = "https://example.com/generated/sample.png"
    metadata = {
        "prompt": payload.prompt,
        "negative_prompt": payload.negative_prompt,
        "width": payload.width,
        "height": payload.height,
        "steps": payload.num_inference_steps,
        "guidance": payload.guidance_scale
    }
    return {"success": True, "image_url": fake_url, "metadata": metadata}


# ================== Save Image Tool ==================
class SaveImageInput(BaseModel):
    image_url: str = Field(..., description="URL of the image to save")
    prompt: str = Field(..., description="Original prompt used to generate the image")

class SavedImageMetadata(BaseModel):
    id: str
    filename: str
    prompt: str
    image_url: str

class SaveImageOutput(BaseModel):
    success: bool
    metadata: SavedImageMetadata

@app.post("/save-image", response_model=SaveImageOutput, summary="Save a generated image locally")
def save_image(payload: SaveImageInput):
    """
    Download and save the image from the URL, generate a unique ID, and return metadata.
    """
    
    return {"success": True, "metadata": ''}


# ================== List Saved Images Tool ==================
class ListSavedImagesOutput(BaseModel):
    success: bool
    images: List[SavedImageMetadata]

@app.get("/list-saved-images", response_model=ListSavedImagesOutput, summary="List all saved images with metadata and thumbnails")
def list_saved_images():
    """
    List all images saved in the local filesystem with metadata.
    """
    
    return {"success": True, "images": []}


mcp = FastMCP.from_fastapi(app=app)

if __name__ == '__main__':
    mcp.run(transport='sse',port=50115)