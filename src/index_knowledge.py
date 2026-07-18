#!/usr/bin/env python3
"""
Knowledge Base Indexing Script

This script indexes all knowledge base data including:
- Knowledge base JSON
- LinkedIn posts JSON
- Websites JSONL

Usage: python3 -m src.index_knowledge (or run inside the project with PYTHONPATH=src)
"""

import asyncio
import os
import sys

from assistant.agent.processor import (
    KnowledgeJsonProcessor,
    LinkedinJsonProcessor,
    WebsiteJsonlProcessor,
)


async def index_knowledge():
    """Index all knowledge base data sources."""
    try:
        print("🔄 Starting knowledge base indexing...")

        # Process knowledge base
        print("📚 Processing knowledge base...")
        knowledge_processor = KnowledgeJsonProcessor()
        knowledge_result = await knowledge_processor.process_data("datasources/knowledge_base.json")
        print(f"✅ Knowledge base indexed: {knowledge_result}")

        # Process LinkedIn posts
        print("💼 Processing LinkedIn posts...")
        linkedin_processor = LinkedinJsonProcessor()
        linkedin_result = await linkedin_processor.process_data("datasources/posts.json")
        print(f"✅ LinkedIn posts indexed: {linkedin_result}")

        # Process websites
        print("🌐 Processing websites...")
        website_processor = WebsiteJsonlProcessor()
        websites_result = await website_processor.process_data("datasources/websites.jsonl")
        print(f"✅ Websites indexed: {websites_result}")

        print("🎉 All knowledge indexed successfully!")

    except Exception as e:
        print(f"❌ Error indexing knowledge: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(index_knowledge())
