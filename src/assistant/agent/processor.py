import json
import os
from typing import Any, Dict, List

from assistant.services.vector_store import Record, VectorStore


async def index_investment_knowledge(json_path: str) -> Dict[str, Any]:
    """Load *json_path* and index all investment topics into ChromaDB.

    Args:
        json_path: Path to investment_kb.json (absolute or relative to CWD).

    Returns:
        Dict with ``status`` and ``message`` keys.
    """
    # Load JSON
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {"status": "error", "message": f"File not found: {json_path}"}
    except json.JSONDecodeError as exc:
        return {"status": "error", "message": f"Invalid JSON in {json_path}: {exc}"}

    items = data if isinstance(data, list) else data.get("topics", [])
    if not items:
        return {"status": "warning", "message": f"No topics found in {json_path}"}

    # Build records
    records: List[Record] = []
    for idx, item in enumerate(items):
        doc_id = item.get("id") or f"inv_{idx}"
        topic = item.get("topic") or item.get("title") or "Investice"
        content = item.get("content") or item.get("description") or ""
        category = item.get("category") or "general"

        formatted_text = f"Téma: {topic}\nKategorie: {category}\nObsah: {content}"
        records.append(
            Record(
                doc_id=doc_id,
                knowledge=[formatted_text],
                metadata=[
                    {
                        "source": json_path,
                        "topic": topic,
                        "category": category,
                        "id": doc_id,
                    }
                ],
            )
        )

    # Store into vector DB
    try:
        store = VectorStore()
        count = await store.store_embeddings(records)
        return {
            "status": "success",
            "message": f"Successfully indexed {count} items into vector store",
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


# ---------------------------------------------------------------------------
# Backward-compatibility shim used by api_server.py
# ---------------------------------------------------------------------------


class InvestmentKnowledgeProcessor:
    """Thin wrapper so existing call-sites (api_server, index_knowledge.py) work."""

    async def process_data(self, json_path: str) -> Dict[str, Any]:
        """Delegate to module-level function."""
        return await index_investment_knowledge(json_path)
