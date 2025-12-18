from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from fastmcp import FastMCP

app = FastAPI(title="GitLab Tool MCP Service", version="1.0.0")

# ===== Models =====

class CreateOrUpdateFileInput(BaseModel):
    project_id: str = Field(..., description="Project ID or URL-encoded path")
    file_path: str = Field(..., description="Path where to create/update the file")
    content: str = Field(..., description="Content of the file")
    commit_message: str = Field(..., description="Commit message")
    branch: str = Field(..., description="Branch to create/update the file in")
    previous_path: Optional[str] = Field(None, description="Path of the file to move/rename")

class CreateOrUpdateFileOutput(BaseModel):
    success: bool
    file_content: Dict[str, Any] = Field(..., description="File content details")
    commit_details: Dict[str, Any] = Field(..., description="Commit details")

class PushFilesInput(BaseModel):
    project_id: str
    branch: str
    files: List[Dict[str, str]]  # each file: {"file_path": str, "content": str}
    commit_message: str

class PushFilesOutput(BaseModel):
    success: bool
    updated_ref: str

class SearchRepositoriesInput(BaseModel):
    search: str
    page: Optional[int] = 1
    per_page: Optional[int] = 20

class SearchRepositoriesOutput(BaseModel):
    success: bool
    results: List[Dict[str, Any]]

class CreateRepositoryInput(BaseModel):
    name: str
    description: Optional[str] = None
    visibility: Optional[str] = Field("private", description="private, internal or public")
    initialize_with_readme: Optional[bool] = False

class CreateRepositoryOutput(BaseModel):
    success: bool
    project_details: Dict[str, Any]

class GetFileContentsInput(BaseModel):
    project_id: str
    file_path: str
    ref: Optional[str] = None

class GetFileContentsOutput(BaseModel):
    success: bool
    contents: Any

class CreateIssueInput(BaseModel):
    project_id: str
    title: str
    description: Optional[str] = None
    assignee_ids: Optional[List[int]] = None
    labels: Optional[List[str]] = None
    milestone_id: Optional[int] = None

class CreateIssueOutput(BaseModel):
    success: bool
    issue_details: Dict[str, Any]

class CreateMergeRequestInput(BaseModel):
    project_id: str
    title: str
    description: Optional[str] = None
    source_branch: str
    target_branch: str
    draft: Optional[bool] = False
    allow_collaboration: Optional[bool] = False

class CreateMergeRequestOutput(BaseModel):
    success: bool
    mr_details: Dict[str, Any]

class ForkRepositoryInput(BaseModel):
    project_id: str
    namespace: Optional[str] = None

class ForkRepositoryOutput(BaseModel):
    success: bool
    forked_project_details: Dict[str, Any]

class CreateBranchInput(BaseModel):
    project_id: str
    branch: str
    ref: Optional[str] = None

class CreateBranchOutput(BaseModel):
    success: bool
    branch_ref: str


# ===== API =====

@app.post("/create-or-update-file", response_model=CreateOrUpdateFileOutput,summary="Create or update a file.")
def create_or_update_file(payload: CreateOrUpdateFileInput):
    return {
        "success": True,
        "file_content": {"file_path": payload.file_path, "content_preview": payload.content[:20]},
        "commit_details": {"commit_message": payload.commit_message, "branch": payload.branch}
    }

@app.post("/push-files", response_model=PushFilesOutput,summary="Push files to a specific branch")
def push_files(payload: PushFilesInput):
    return {"success": True, "updated_ref": f"refs/heads/{payload.branch}"}

@app.post("/search-repositories", response_model=SearchRepositoriesOutput,summary="Search code repositories")
def search_repositories(payload: SearchRepositoriesInput):
    return {
        "success": True,
        "results": [
            {"project_id": "123", "name": f"Project matching {payload.search}", "description": "Fake project"}
        ]
    }

@app.post("/create-repository", response_model=CreateRepositoryOutput,summary="Create a new code repository")
def create_repository(payload: CreateRepositoryInput):
    return {
        "success": True,
        "project_details": {
            "id": "new_proj_001",
            "name": payload.name,
            "description": payload.description or "",
            "visibility": payload.visibility,
            "readme_initialized": payload.initialize_with_readme
        }
    }

@app.post("/get-file-contents", response_model=GetFileContentsOutput,summary="Get file contents")
def get_file_contents(payload: GetFileContentsInput):
    return {
        "success": True,
        "contents": {"file_path": payload.file_path, "ref": payload.ref or "main", "content": "Fake file content"}
    }

@app.post("/create-issue", response_model=CreateIssueOutput,summary="Create a new issue")
def create_issue(payload: CreateIssueInput):
    return {
        "success": True,
        "issue_details": {
            "id": 456,
            "title": payload.title,
            "description": payload.description or "",
            "assignees": payload.assignee_ids or [],
            "labels": payload.labels or [],
            "milestone_id": payload.milestone_id
        }
    }

@app.post("/create-merge-request", response_model=CreateMergeRequestOutput,summary="Create a new merge request")
def create_merge_request(payload: CreateMergeRequestInput):
    return {
        "success": True,
        "mr_details": {
            "id": 789,
            "title": payload.title,
            "source_branch": payload.source_branch,
            "target_branch": payload.target_branch,
            "draft": payload.draft,
            "allow_collaboration": payload.allow_collaboration
        }
    }

@app.post("/fork-repository", response_model=ForkRepositoryOutput,summary="Create a fork of a repository")
def fork_repository(payload: ForkRepositoryInput):
    return {
        "success": True,
        "forked_project_details": {
            "id": "fork_001",
            "original_project": payload.project_id,
            "namespace": payload.namespace or "default"
        }
    }

@app.post("/create-branch", response_model=CreateBranchOutput,summary="Create a new branch")
def create_branch(payload: CreateBranchInput):
    return {
        "success": True,
        "branch_ref": f"refs/heads/{payload.branch}"
    }


# ===== MCP 启动 =====

mcp = FastMCP.from_fastapi(app=app)

if __name__ == '__main__':
    mcp.run(transport='sse', port=50137)
