from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import sys
import os
from typing import Dict, Any
import pathlib
from src.chat import AIService
from src.vector_store import VectorStore
from agent.processor import KnowledgeJsonProcessor
from json import JSONDecodeError


# Import your processor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create FastAPI app
app = FastAPI(title="Knowledge Base API", 
              description="API for accessing knowledge base data",
              version="1.0.0")

# Set up templates
templates = Jinja2Templates(directory="templates")

# Make sure static directory exists
static_dir = pathlib.Path("static")
if not static_dir.exists():
    static_dir.mkdir(exist_ok=True)

# Mount static files with the correct path
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Create a single instance of AIService to be reused
ai_service = AIService()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the home page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat", response_class=JSONResponse)
async def chat(request: Request) -> Dict[str, Any]:
    """
    Chat endpoint compatible with DeepChat
    
    Args:
        request: The incoming request with DeepChat format
        
    Returns:
        JSON response in DeepChat-compatible format
    """
    try:
        # Parse the incoming JSON request
        try:
            data = await request.json()
        except JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON")
        
        # Extract the message from DeepChat request format
        # DeepChat typically sends messages in an array format
        messages = data.get("messages", [])
        
        if not messages:
            return {"message": "No messages provided"}
        
        # Get the last message content
        last_message = messages[-1].get("content", "")
        
        # Process the message using your AI service - make sure to await the response
        response = await ai_service.chat(last_message)
        
        # Return in DeepChat-compatible format
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": response
                }
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/knowledge_base", response_class=JSONResponse)
async def get_knowledge_base() -> Dict[str, Any]:
    """
    Retrieve knowledge base data
    
    Returns:
        JSON data from the knowledge base
    """
    try:
        store = VectorStore()
        all_docs = await store.get_all_records()
        return {"message": all_docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/index_knowledge", response_class=JSONResponse)
async def get_knowledge_base() -> Dict[str, Any]:
    """
    Retrieve knowledge base data
    
    Returns:
        JSON data from the knowledge base
    """
    try:
        processor = KnowledgeJsonProcessor()

        indexing = await processor.store_data("datasources/knowledge_base.json")
        return indexing
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))    

# Run the app with Uvicorn if executed as main script
if __name__ == "__main__":
    uvicorn.run("api-server:app", host="0.0.0.0", port=5000, log_level="info")