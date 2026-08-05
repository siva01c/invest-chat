import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from assistant.services.vector_store import Record, VectorStore


class DataProcessor(ABC):
    @abstractmethod
    async def process_data(self, file_path: str):
        pass


class JsonProcessor(DataProcessor):
    """Base class for JSON data processing."""

    def _load_data(self, json_path: str) -> Any:
        """Load JSON data from file."""
        if not os.path.isabs(json_path):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(current_dir, "..", "data")
            json_path = os.path.join(data_dir, json_path)

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {json_path}: {e}")
            return None

    async def process_data(self, json_path: str) -> Dict[str, Any]:
        store = VectorStore()
        records = await self._prepare_data(json_path)
        if not records:
            return {"status": "warning", "message": f"No records processed from {json_path}"}
        try:
            count = await store.store_embeddings(records)
            return {"status": "success", "message": f"Successfully indexed {count} items into vector store"}
        except Exception as e:
            print(f"Error creating embeddings: {e}")
            return {"status": "error", "message": str(e)}

    @abstractmethod
    async def _prepare_data(self, json_path: str) -> List[Record]:
        pass


class InvestmentKnowledgeProcessor(JsonProcessor):
    """Specialized processor for investment knowledge base JSON files."""

    async def _prepare_data(self, json_path: str) -> List[Record]:
        data = self._load_data(json_path)
        if not data:
            return []

        records = []
        items = data if isinstance(data, list) else data.get("topics", [])
        for idx, item in enumerate(items):
            doc_id = item.get("id") or f"inv_{idx}"
            topic = item.get("topic") or item.get("title") or "Investice"
            content = item.get("content") or item.get("description") or ""
            category = item.get("category") or "general"

            formatted_text = f"Téma: {topic}\nKategorie: {category}\nObsah: {content}"
            record = Record(
                doc_id=doc_id,
                knowledge=[formatted_text],
                metadata=[{
                    "source": json_path,
                    "topic": topic,
                    "category": category,
                    "id": doc_id,
                }]
            )
            records.append(record)

        return records
