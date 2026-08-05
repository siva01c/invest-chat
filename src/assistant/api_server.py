"""FastAPI application for Investment RAG Chatbot."""

import os
import secrets
from pathlib import Path

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from assistant.agent.processor import InvestmentKnowledgeProcessor
from assistant.api.routes import chat

app = FastAPI(
    title="Invest Chat AI Assistant",
    description="Vzdělávací RAG asistent pro osobní finance a investování",
    version="2.0.0",
)

# ---------------------------------------------------------------------------
# CORS — origins from env, no wildcard with credentials
# ---------------------------------------------------------------------------
_cors_origins_raw = os.getenv("CORS_ORIGINS", "http://localhost:8000")
CORS_ORIGINS = [o.strip() for o in _cors_origins_raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,   # Cannot combine credentials with wildcard
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-Session-Id", "X-API-Key"],
)

# ---------------------------------------------------------------------------
# Templates & Static files
# ---------------------------------------------------------------------------
templates_dir = Path("templates")
templates_dir.mkdir(exist_ok=True)
templates = Jinja2Templates(directory=str(templates_dir))

static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# ---------------------------------------------------------------------------
# Include routers
# ---------------------------------------------------------------------------
app.include_router(chat.router)

# ---------------------------------------------------------------------------
# Admin API-key guard for management endpoints
# ---------------------------------------------------------------------------
_admin_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _require_admin_key(api_key: str | None = Security(_admin_api_key_header)) -> None:
    """Dependency that enforces the ADMIN_API_KEY environment variable."""
    expected = os.getenv("ADMIN_API_KEY", "")
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="ADMIN_API_KEY is not configured on this server.",
        )
    if not api_key or not secrets.compare_digest(api_key, expected):
        raise HTTPException(status_code=403, detail="Invalid or missing API key.")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index_page(request: Request):
    """Render main chat UI page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "app": "Invest Chat AI Assistant", "version": "2.0.0"}


@app.post("/api/index", dependencies=[Depends(_require_admin_key)])
async def reindex_knowledge():
    """Re-index investment knowledge base into ChromaDB.

    Requires the ``X-API-Key`` header with the value of ``ADMIN_API_KEY``.
    """
    processor = InvestmentKnowledgeProcessor()
    kb_path = os.path.join("datasources", "investment_kb.json")
    result = await processor.process_data(kb_path)
    return JSONResponse(content=result)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    reload = os.getenv("RELOAD", "false").lower() == "true"
    uvicorn.run("assistant.api_server:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    main()
