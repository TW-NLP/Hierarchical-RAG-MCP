from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from fastmcp import FastMCP

app = FastAPI(title="ArXiv Paper Management API", version="1.0.0")

# Directory to store downloaded papers
PAPER_DIR = "downloaded_papers"

# ================== Search Papers ==================
class SearchPapersInput(BaseModel):
    query: str = Field(..., description="Search query string for papers")
    max_results: Optional[int] = Field(10, description="Maximum number of results to return")
    date_from: Optional[str] = Field(None, description="Filter results published after this date (YYYY-MM-DD)")
    categories: Optional[List[str]] = Field(None, description="List of arXiv categories to filter")

class PaperSummary(BaseModel):
    paper_id: str
    title: str
    authors: List[str]
    abstract: str
    published_date: str
    categories: List[str]

class SearchPapersOutput(BaseModel):
    success: bool
    results: List[PaperSummary]

@app.post("/search-papers", response_model=SearchPapersOutput,summary="Search for papers with filters like query, date_from, categories")
def search_papers(payload: SearchPapersInput):
    """Search for papers with filters like query, date_from, categories"""
    # TODO: integrate with arXiv search API
    return {"success": True, "results": ['Paper 1', 'Paper 2', 'Paper 3']}

# ================== Download Paper ==================
class DownloadPaperInput(BaseModel):
    paper_id: str = Field(..., description="arXiv paper ID, e.g., '2401.12345'")

class DownloadPaperOutput(BaseModel):
    success: bool
    path: str  # Local file path of downloaded PDF

@app.post("/download-paper", response_model=DownloadPaperOutput,summary="Download a paper by its arXiv ID to local storage")
def download_paper(payload: DownloadPaperInput):
    """Download a paper by its arXiv ID to local storage"""
    # TODO: download PDF and save under PAPER_DIR
    fake_path = f"{PAPER_DIR}/{payload.paper_id}.pdf"
    return {"success": True, "path": fake_path}

# ================== List Downloaded Papers ==================
class ListPapersOutput(BaseModel):
    success: bool
    papers: List[str]  # List of paper IDs or filenames

@app.get("/list-papers", response_model=ListPapersOutput,summary="List all downloaded papers in local storage")
def list_papers():
    """List all downloaded papers in local storage"""
    # TODO: scan PAPER_DIR for PDFs
    return {"success": True, "papers": []}

# ================== Read Paper ==================
class ReadPaperInput(BaseModel):
    paper_id: str = Field(..., description="arXiv paper ID of a previously downloaded paper")

class ReadPaperOutput(BaseModel):
    success: bool
    content: str  # Extracted text or metadata from PDF

@app.post("/read-paper", response_model=ReadPaperOutput,summary="Read and return the contents of a downloaded paper")
def read_paper(payload: ReadPaperInput):
    """Read and return the contents of a downloaded paper"""
    # TODO: read PDF from local storage and extract text
    return {"success": True, "content": "extracted content from the paper"}


mcp = FastMCP.from_fastapi(app=app)

if __name__ == '__main__':
    mcp.run(transport='sse',port=50122)