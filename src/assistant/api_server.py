"""FastAPI application for Investment RAG Chatbot."""

import os
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

from assistant.api.routes import chat
from assistant.agent.processor import InvestmentKnowledgeProcessor

app = FastAPI(
    title="Invest Chat AI Assistant",
    description="Vzdělávací RAG asistent pro osobní finance a investování",
    version="2.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup Templates & Static files
templates_dir = Path("templates")
if not templates_dir.exists():
    templates_dir.mkdir(exist_ok=True)
templates = Jinja2Templates(directory=str(templates_dir))

static_dir = Path("static")
if not static_dir.exists():
    static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Include chat router
app.include_router(chat.router)


@app.get("/", response_class=HTMLResponse)
async def index_page(request: Request):
    """Render main chat UI page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "app": "Invest Chat AI Assistant", "version": "2.0.0"}


@app.post("/api/index")
async def reindex_knowledge():
    """Re-index investment knowledge base into ChromaDB."""
    processor = InvestmentKnowledgeProcessor()
    kb_path = os.path.join("datasources", "investment_kb.json")
    result = await processor.process_data(kb_path)
    return JSONResponse(content=result)


def main():
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("assistant.api_server:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    main()
