from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from fastmcp import FastMCP

#URL=https://www.modelscope.cn/mcp/servers/@Rudra-ravi/wikipedia-mcp
wiki_app = FastAPI(title="Wikipedia Tools API", version="1.0.0")


# ================== Wikipedia Tools ==================

class SearchWikipediaInput(BaseModel):
    query: str
    limit: Optional[int] = 10

@wiki_app.post("/search_wikipedia", summary="Search Wikipedia for articles")
def search_wikipedia(payload: SearchWikipediaInput):
    """
    根据查询关键词搜索 Wikipedia 文章。
    返回匹配的文章标题、摘要片段和元数据。
    """
    return {"success": True, "data": ['Article 1', 'Article 2', 'Article 3']}


class TitleInput(BaseModel):
    title: str


@wiki_app.post("/get_article", summary="Get full content of a Wikipedia article")
def get_article(payload: TitleInput):
    """
    获取指定 Wikipedia 文章的完整内容，包括正文、摘要、章节、链接和分类信息。
    """
    return {"success": True,'data': {
        "title": payload.title,
        "content": "Full content of the article goes here.",
        "summary": "Brief summary of the article.",
        "sections": ["Section 1", "Section 2", "Section 3"],
        "links": ["Link 1", "Link 2", "Link 3"],
        "categories": ["Category 1", "Category 2"]
    }
    }


@wiki_app.post("/get_summary", summary="Get summary of a Wikipedia article")
def get_summary(payload: TitleInput):
    """
    获取 Wikipedia 文章的简要摘要。
    """
    return {"success": True, "data": {
        "title": payload.title,
        "summary": "Brief summary of the article."
    }}


@wiki_app.post("/get_sections", summary="Get sections of a Wikipedia article")
def get_sections(payload: TitleInput):
    """
    获取 Wikipedia 文章的结构化章节内容。
    """
    return {"success": True}


@wiki_app.post("/get_links", summary="Get links in a Wikipedia article")
def get_links(payload: TitleInput):
    """
    获取 Wikipedia 文章中包含的链接（指向其他 Wikipedia 文章）。
    """
    return {"success": True}


class RelatedTopicsInput(BaseModel):
    title: str
    limit: Optional[int] = 10

@wiki_app.post("/get_related_topics", summary="Get related topics from Wikipedia")
def get_related_topics(payload: RelatedTopicsInput):
    """
    基于链接和分类信息，获取与指定 Wikipedia 文章相关的主题列表。
    """
    return {"success": True}

mcp = FastMCP.from_fastapi(app=wiki_app)

if __name__ == '__main__':
    mcp.run(transport='sse',port=50129)