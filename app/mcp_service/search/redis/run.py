from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Union, List
from fastmcp import FastMCP

app = FastAPI(title="Redis Tools API", version="1.0.0")


# ============ Data Models ============

class SetKeyInput(BaseModel):
    key: str = Field(..., description="Redis key")
    value: str = Field(..., description="Value to store")
    expireSeconds: Optional[int] = Field(None, description="Expiration time in seconds")

class GetKeyInput(BaseModel):
    key: str = Field(..., description="Redis key to retrieve")

class DeleteKeyInput(BaseModel):
    key: Union[str, List[str]] = Field(..., description="Single key or list of keys to delete")

class ListKeysInput(BaseModel):
    pattern: Optional[str] = Field("*", description="Pattern to match keys (default '*')")


# ============ Endpoints ============

@app.post("/set",summary="Set a Redis key-value pair with optional expiration")
def set_key(payload: SetKeyInput):
    """
    Set a Redis key-value pair with optional expiration.
    """
    # TODO: Replace with actual Redis logic
    return {
        "success": True,
        "message": f"Key '{payload.key}' set with value '{payload.value}'",
        "expireSeconds": payload.expireSeconds
    }


@app.post("/get", summary="Get value by key from Redis")
def get_key(payload: GetKeyInput):
    """
    Get value by key from Redis.
    """
    # TODO: Replace with actual Redis logic
    return {
        "success": True,
        "key": payload.key,
        "value": "mocked_value"
    }


@app.post("/delete", summary="Delete one or more keys from Redis")
def delete_key(payload: DeleteKeyInput):
    """
    Delete one or more keys from Redis.
    """
    # TODO: Replace with actual Redis logic
    keys = [payload.key] if isinstance(payload.key, str) else payload.key
    return {
        "success": True,
        "deleted_keys": keys
    }


@app.post("/list", summary="List Redis keys matching a pattern")
def list_keys(payload: ListKeysInput):
    """
    List Redis keys matching a pattern.
    """
    # TODO: Replace with actual Redis logic
    return {
        "success": True,
        "pattern": payload.pattern,
        "keys": ["mock:key1", "mock:key2"]
    }
mcp = FastMCP.from_fastapi(app=app)

if __name__ == '__main__':
    mcp.run(transport='sse', port=50132)