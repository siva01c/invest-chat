import json
from abc import ABC, abstractmethod

from assistant.services.vector_store import Record, VectorStore


class DataProcessor(ABC):
    @abstractmethod
    async def process_data(self):
        pass


class JsonProcessor(DataProcessor):
    """Base class for JSON processing"""

    def _load_data(self, json_path: str):
        """Load JSON data from file"""
        import os

        # Handle relative paths by looking in the data directory
        if not os.path.isabs(json_path):
            # Get the directory where this module is located
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up to assistant directory, then to data
            data_dir = os.path.join(current_dir, "..", "data")
            json_path = os.path.join(data_dir, json_path)

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: File {json_path} not found.")
            return None
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {json_path}.")
            return None

    async def _prepare_data(self, json_path: str):
        """Generic process method to be overridden"""
        raise NotImplementedError("Subclasses must implement process_data")

    async def process_data(self, json_path: str):
        store = VectorStore()
        records = await self._prepare_data(json_path)
        try:
            result = await store.store_embeddings(records)
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error creating embeddings: {str(e)}")

        return {"message": "Data stored"}


class KnowledgeJsonProcessor(JsonProcessor):
    """Specialized processor for knowledge base JSON files"""

    async def _prepare_data(self, json_path: str):
        data = self._load_data(json_path)
        if not data:
            return []

        records = []
        for idx, topic in enumerate(data.get("topics", [])):
            record = Record()
            document_parts = [
                f"Title: {topic.get('title', '')}",
                f"Description: {topic.get('description', '')}",
            ]

            if "key_competences" in topic:
                competences = "\n".join([f"- {comp}" for comp in topic["key_competences"]])
                document_parts.append(f"Key Competences:\n{competences}")

            if "keywords" in topic:
                keywords = "\n".join([f"- {keyword}" for keyword in topic["keywords"]])
                document_parts.append(f"Keywords:\n{keywords}")

            document = "\n\n".join(document_parts)
            record.knowledge.append(document)

            record.metadata.append(
                {
                    "source": json_path,
                    "index": idx,
                    "title": topic.get("title", ""),
                    "type": "knowledge_base",
                }
            )
            record.id = f"topic_{idx}"
            records.append(record)

        return records


class LinkedinJsonProcessor(JsonProcessor):
    """Specialized processor for knowledge base JSON files"""

    async def _prepare_data(self, json_path: str):
        data = self._load_data(json_path)
        if not data:
            return []

        records = []
        # Sort data items by timestamp/id to ensure consistent enumeration
        sorted_topics = sorted(data.items(), key=lambda x: x[0])

        for idx, (post_id, topic) in enumerate(sorted_topics, 1):
            record = Record()
            document_parts = [
                f"Post #{idx}:",
                f"Status: {topic.get('text', '')}",
                f"Description: LinkedIn status by {topic.get('user', '')}, {topic.get('metadata', '')}",
            ]

            document = "\n\n".join(document_parts)
            record.knowledge.append(document)

            record.metadata.append(
                {
                    "source": json_path,
                    "index": idx,
                    "post_id": post_id,
                    "title": f"LinkedIn post #{idx}: {topic.get('user', '')}, {topic.get('metadata', '')}",
                    "type": "linkedin",
                }
            )
            record.id = f"linkedin_post_{idx}"
            records.append(record)

        return records


class WebsiteJsonlProcessor(JsonProcessor):
    """Specialized processor for website JSONL files"""

    def _load_data(self, jsonl_path: str):
        """Load JSONL data from file"""
        import os

        # Handle relative paths by looking in the data directory
        if not os.path.isabs(jsonl_path):
            # Get the directory where this module is located
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up to assistant directory, then to data
            data_dir = os.path.join(current_dir, "..", "data")
            jsonl_path = os.path.join(data_dir, jsonl_path)

        try:
            data = []
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"Error parsing line {line_num}: {e}")
            return data
        except FileNotFoundError:
            print(f"Error: File {jsonl_path} not found.")
            return []

    async def _prepare_data(self, jsonl_path: str):
        data = self._load_data(jsonl_path)
        if not data:
            return []

        records = []
        for idx, website in enumerate(data):
            record = Record()

            # Extract content for embedding
            content_parts = []
            if "title" in website:
                content_parts.append(f"Title: {website['title']}")
            if "url" in website:
                content_parts.append(f"URL: {website['url']}")
            if "fullText" in website:
                content_parts.append(f"Content: {website['fullText']}")

            # Add SEO metadata if available
            if "seo" in website and "metaTags" in website["seo"]:
                meta = website["seo"]["metaTags"]
                if "description" in meta:
                    content_parts.append(f"Description: {meta['description']}")
                if "keywords" in meta:
                    content_parts.append(f"Keywords: {meta['keywords']}")

            document = "\n\n".join(content_parts)
            record.knowledge.append(document)

            record.metadata.append(
                {
                    "source": jsonl_path,
                    "index": idx,
                    "url": website.get("url", ""),
                    "title": website.get("title", ""),
                    "timestamp": website.get("timestamp", ""),
                    "type": "website",
                }
            )
            record.id = f"website_{idx}"
            records.append(record)

        return records


class PDFProcessor(DataProcessor):
    async def process_data(self):
        pass
