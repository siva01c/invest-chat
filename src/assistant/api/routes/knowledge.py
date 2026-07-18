"""Knowledge base endpoints."""

from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from assistant.agent.processor import (
    KnowledgeJsonProcessor,
    LinkedinJsonProcessor,
    WebsiteJsonlProcessor,
)
from assistant.core.config.services import get_vector_store
from assistant.core.models import KnowledgeBaseResponse

router = APIRouter()


@router.get("/knowledge_base", response_class=JSONResponse)
async def get_knowledge_base() -> Dict[str, Any]:
    """
    Retrieve knowledge base data

    Returns:
        JSON data from the knowledge base
    """
    try:
        store = get_vector_store()
        all_docs = await store.get_all_records()
        response = KnowledgeBaseResponse(message=all_docs)
        return response.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/index_knowledge", response_class=JSONResponse)
async def index_knowledge_base() -> Dict[str, Any]:
    """
    Index knowledge base data from various sources

    Returns:
        Status message about indexed data
    """
    try:
        # Process knowledge base
        knowledge_processor = KnowledgeJsonProcessor()
        knowledge_base = await knowledge_processor.process_data("datasources/knowledge_base.json")

        # Process LinkedIn posts
        linkedin_processor = LinkedinJsonProcessor()
        linkedin = await linkedin_processor.process_data("datasources/posts.json")

        # Process websites data
        website_processor = WebsiteJsonlProcessor()
        websites = await website_processor.process_data("datasources/websites.jsonl")

        return {
            "message": f"Data stored \n knowledge_base: {knowledge_base}, linkedin: {linkedin}, websites: {websites}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
