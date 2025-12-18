from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from fastmcp import FastMCP

#URL:https://modelscope.cn/mcp/servers/@modelcontextprotocol/filesystem
api = FastAPI(title="File System Tools API", version="1.0.0")


# ================== File System Operations ==================

class ReadFileInput(BaseModel):
    path: str  # Path to file

@api.post("/read_file", summary="Read complete contents of a file")
def read_file(payload: ReadFileInput):
    """
    Read complete file contents with UTF-8 encoding
    Input: { "path": "/path/to/file.txt" }
    Output: { "success": true, "data": "file contents..." }
    """
    return {"success": True, "data": ""}


class ReadMultipleFilesInput(BaseModel):
    paths: List[str]  # List of file paths

@api.post("/read_multiple_files", summary="Read multiple files simultaneously")
def read_multiple_files(payload: ReadMultipleFilesInput):
    """
    Read multiple files; failures won't stop the operation
    Input: { "paths": ["/a.txt", "/b.txt"] }
    Output: { "success": true, "data": { "/a.txt": "...", "/b.txt": null } }
    """
    return {"success": True, "data": {path: "" for path in payload.paths}}


class WriteFileInput(BaseModel):
    path: str
    content: str

@api.post("/write_file", summary="Create or overwrite a file")
def write_file(payload: WriteFileInput):
    """
    Create new file or overwrite existing
    Input: { "path": "/path/to/file.txt", "content": "..." }
    """
    return {"success": True}


class EditOperation(BaseModel):
    oldText: str
    newText: str
    dryRun: Optional[bool] = Field(False, description="Preview changes without applying")

class EditFileInput(BaseModel):
    path: str
    edits: List[EditOperation]

@api.post("/edit_file", summary="Make selective edits using advanced pattern matching")
def edit_file(payload: EditFileInput):
    """
    Perform line/multi-line edits with indentation and diff preview
    Input: { "path": "/file.py", "edits": [{"oldText": "foo", "newText": "bar", "dryRun": true}] }
    Output: diff or success
    """
    return {"success": True, "diff": []}


class CreateDirectoryInput(BaseModel):
    path: str

@api.post("/create_directory", summary="Create new directory or ensure it exists")
def create_directory(payload: CreateDirectoryInput):
    """
    Creates parent directories if needed; silent if exists
    """
    return {"success": True}


class ListDirectoryInput(BaseModel):
    path: str

@api.post("/list_directory", summary="List directory contents")
def list_directory(payload: ListDirectoryInput):
    """
    List contents with [FILE]/[DIR] prefixes
    Output: { "success": true, "data": ["[DIR] subfolder", "[FILE] file.txt"] }
    """
    return {"success": True, "data": []}


class MoveFileInput(BaseModel):
    source: str
    destination: str

@api.post("/move_file", summary="Move or rename files and directories")
def move_file(payload: MoveFileInput):
    """
    Move or rename; fails if destination exists
    """
    return {"success": True}


class SearchFilesInput(BaseModel):
    path: str
    pattern: str
    excludePatterns: Optional[List[str]] = []

@api.post("/search_files", summary="Recursively search for files/directories")
def search_files(payload: SearchFilesInput):
    """
    Case-insensitive glob search with exclusions
    Output: { "success": true, "data": ["/path/match1", "/path/match2"] }
    """
    return {"success": True, "data": []}


class FileInfo(BaseModel):
    size: int
    created: str
    modified: str
    accessed: str
    type: str
    permissions: str

class GetFileInfoInput(BaseModel):
    path: str

@api.post("/get_file_info", summary="Get file or directory metadata")
def get_file_info(payload: GetFileInfoInput):
    """
    Returns size, timestamps, type, permissions
    """
    return {"success": True, "data": FileInfo(size=0, created="", modified="", accessed="", type="file", permissions="").dict()}


@api.get("/list_allowed_directories", summary="List allowed directories")
def list_allowed_directories():
    """
    List server-accessible directories without input
    Output: { "success": true, "data": ["/home/user", "/tmp"] }
    """
    return {"success": True, "data": []}

mcp = FastMCP.from_fastapi(app=api)

if __name__ == '__main__':
    mcp.run(transport='sse',port=50113)