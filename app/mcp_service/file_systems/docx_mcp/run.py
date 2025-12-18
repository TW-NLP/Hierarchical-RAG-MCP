from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict
from fastmcp import FastMCP

# URL:https://modelscope.cn/mcp/servers/@famano/mcp-server-office
app = FastAPI(title="Docx Tools API", version="1.0.0")


# ================== Available Tools ==================

class PathInput(BaseModel):
    path: str

@app.post("/read_docx", summary="Read complete contents of a docx file including tables and images")
def read_docx(payload: PathInput):
    """
    Input: path (string) - Absolute path to the target file
    Note: Images are converted to [Image] placeholders, and track changes are not shown
    """
    return {"success": True,'data': {
        "title": "Document Title",
        "content": "This is the main content of the document. It can include paragraphs, lists, and other text elements.",
        "tables": [
            {
                "rows": [
                    ["Header 1", "Header 2", "Header 3"],
                    ["Row 1 Col 1", "Row 1 Col 2", "Row 1 Col 3"],
                    ["Row 2 Col 1", "Row 2 Col 2", "Row 2 Col 3"]
                ],
                "caption": "This is a sample table caption."
            }
        ]
    }}

class WriteDocxInput(BaseModel):
    path: str
    content: str

@app.post("/write_docx", summary="Create a new docx file with given content")
def write_docx(payload: WriteDocxInput):
    """
    Input:
      path (string) - Absolute path to target file
      content (string) - Content to write to the file
    Note: Use double line breaks for new paragraphs, and [Table] tag with | separators for tables
    """
    return {"success": True}


class EditDocxParagraphItem(BaseModel):
    paragraph_index: int
    search: str
    replace: str

class EditDocxParagraphInput(BaseModel):
    path: str
    edits: List[EditDocxParagraphItem]

@app.post("/edit_docx_paragraph", summary="Make text replacements in specified paragraphs of a docx file")
def edit_docx_paragraph(payload: EditDocxParagraphInput):
    """
    Input:
      path (string) - Absolute path to file to edit
      edits (array) - List of dictionaries containing search/replace text and paragraph index
        paragraph_index (number) - 0-based index of the paragraph to edit
        search (string) - Text to find within the specified paragraph
        replace (string) - Text to replace with
    Note: Each search string must match exactly once within the specified paragraph
    """
    return {"success": True}


class InsertItem(BaseModel):
    text: str
    paragraph_index: Optional[int] = None

class EditDocxInsertInput(BaseModel):
    path: str
    inserts: List[InsertItem]

@app.post("/edit_docx_insert", summary="Insert new paragraphs into a docx file")
def edit_docx_insert(payload: EditDocxInsertInput):
    """
    Input:
      path (string) - Absolute path to file to edit
      inserts (array) - List of dictionaries containing text and optional paragraph index
        text (string) - Text to insert as a new paragraph
        paragraph_index (number, optional) - 0-based index of the paragraph before which to insert. If not specified, insert at the end.
    """
    return {"success": True}

mcp = FastMCP.from_fastapi(app=app)

if __name__ == '__main__':
    mcp.run(transport='sse',port=50111)