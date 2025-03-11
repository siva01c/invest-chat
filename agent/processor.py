from abc import ABC, abstractmethod
from src.vector_store import VectorStore, Record
import json

class DataProcessor(ABC):
    @abstractmethod
    def _load_data(self, file_path):
        pass

    @abstractmethod
    async def _prepare_data(self):
        pass

    @abstractmethod
    async def store_data(self):
        pass


class JsonProcessor(DataProcessor):
    """Base class for JSON processing"""
    def _load_data(self, json_path: str):
        """Load JSON data from file"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
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
    
    async def store_data(self):
        """Store processed data"""
        raise NotImplementedError("Subclasses must implement store_data")

class KnowledgeJsonProcessor(JsonProcessor):
    """Specialized processor for knowledge base JSON files"""
    async def _prepare_data(self, json_path: str):
        data =  self._load_data(json_path)
        if not data:
            return []

        records = []
        for idx, topic in enumerate(data.get('topics', [])):
            record = Record()
            document_parts = [
                f"Title: {topic.get('title', '')}",
                f"Description: {topic.get('description', '')}"
            ]
            
            if 'key_competences' in topic:
                competences = '\n'.join([f"- {comp}" for comp in topic['key_competences']])
                document_parts.append(f"Key Competences:\n{competences}")
                
            if 'keywords' in topic:
                keywords = '\n'.join([f"- {keyword}" for keyword in topic['keywords']])
                document_parts.append(f"Keywords:\n{keywords}")
                
            document = '\n\n'.join(document_parts)
            record.knowledge.append(document) 
            
            record.metadata.append({
                "source": json_path,
                "index": idx,
                "title": topic.get('title', ''),
                "type": "knowledge_base"
            })
            record.id = f"topic_{idx}"
            records.append(record)

        return records
    
    async def store_data(self, json_path: str):
        store = VectorStore()
        records = await self._prepare_data("datasources/knowledge_base.json")
        try:
            result = await store.store_embeddings(records)
            print(f"Data stored")
        except Exception as e:
            print(f"Error creating embeddings: {str(e)}")
               
        return {"message": "Data stored"}


class PDFProcessor(DataProcessor):
    def _load_data(self, file_path):
        pass
    
    async def process_data(self):
        pass

    async def store_data(self):
        pass