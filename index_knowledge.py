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

from assistant.agent.processor import InvestmentKnowledgeProcessor  # noqa: E402


async def index_knowledge() -> None:
    """Index investment knowledge base data source."""
    kb_path = os.path.join("datasources", "investment_kb.json")

    # Fail-fast: check the file exists before starting expensive operations.
    if not os.path.exists(kb_path):
        print(f"❌ Knowledge base file not found: {kb_path}")
        print("   Make sure you are running this script from the project root")
        print("   and that datasources/investment_kb.json exists.")
        sys.exit(1)

    try:
        print("🔄 Starting investment knowledge base indexing...")
        processor = InvestmentKnowledgeProcessor()
        result = await processor.process_data(kb_path)
        print(f"✅ Indexing result: {result}")
        if result.get("status") == "success":
            print("🎉 Investment knowledge base indexed successfully!")
        else:
            print(f"⚠️  Indexing finished with status: {result.get('status')}")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Error indexing investment knowledge: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(index_knowledge())
