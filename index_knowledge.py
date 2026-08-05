#!/usr/bin/env python3
"""
Investment Knowledge Base Indexing Script

Usage: python3 index_knowledge.py
"""

import asyncio
import os
import sys

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from assistant.agent.processor import InvestmentKnowledgeProcessor


async def index_knowledge():
    """Index investment knowledge base data source."""
    try:
        print("🔄 Starting investment knowledge base indexing...")
        processor = InvestmentKnowledgeProcessor()
        kb_path = os.path.join("datasources", "investment_kb.json")
        result = await processor.process_data(kb_path)
        print(f"✅ Indexing result: {result}")
        print("🎉 Investment knowledge base indexed successfully!")

    except Exception as e:
        print(f"❌ Error indexing investment knowledge: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(index_knowledge())
