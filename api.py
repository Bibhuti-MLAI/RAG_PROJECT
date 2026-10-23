"""
api.py
======
FastAPI web server for the Legal Document Assistant.

It serves the chat page (index.html) and exposes a small REST API on top of
the RAG engine in simple_rag.py.

Run it with:
    uvicorn api:app --host 0.0.0.0 --port 8000

Configuration (environment variables, or put them in `my.env`):
    TOGETHER_API_KEY   API key for Together AI (required to answer questions)
    TOGETHER_MODEL     LLM to use (default: meta-llama/Llama-3.3-70B-Instruct-Turbo)
    TESSERACT_CMD      Full path to tesseract.exe, if it is not on PATH
    ALLOWED_ORIGINS    Comma-separated list of other websites allowed to call
                       this API (not needed when using the built-in page)

Notes on the endpoint functions:
* They are plain `def`, not `async def`. FastAPI runs plain functions in a
  worker thread, so a slow upload or LLM call does not freeze the server for
  other users.
* Errors are returned as proper HTTP status codes with a `detail` message,
  which the web page shows to the user.
"""

import logging
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Resolve every path from this file's folder, so the server works no matter
# which directory it is started from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load settings (such as TOGETHER_API_KEY) from my.env. Real environment
# variables win over the file.
load_dotenv(os.path.join(BASE_DIR, "my.env"))

from simple_rag import SimpleLegalRAG  # noqa: E402  (needs the settings above)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("api")

SESSION_COOKIE_NAME = "legal_rag_session"
SESSION_EXPIRY_DAYS = 7

# The RAG engine, created once when the server starts (see lifespan below).
legal_rag: Optional[SimpleLegalRAG] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start-up: build the RAG engine. If it fails, the server still starts
    and /api/system/status explains what went wrong."""
    global legal_rag
    try:
        engine = SimpleLegalRAG(
            docs_dir=os.path.join(BASE_DIR, "legal_documents"),
            persist_dir=os.path.join(BASE_DIR, "legal_vectordb"),
        )
        engine.setup()
        legal_rag = engine
        logger.info("Legal RAG system initialized successfully")
    except Exception:
        logger.exception("Failed to initialize the Legal RAG system")
        legal_rag = None
    yield


app = FastAPI(title="Legal Support RAG API with OCR", lifespan=lifespan)

# The built-in page is served from this same server, so cross-origin access is
# only needed if another website must call the API. It is off by default.
allowed_origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
if allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str
    # Kept so older clients that send it don't break, but it is ignored: the
    # session always comes from the cookie, so nobody can read another
    # person's conversation by guessing its id.
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[Dict[str, str]]
    session_id: str


class DocumentResponse(BaseModel):
    filename: str
    status: str
    message: str
    converted_from_image: bool = False
    processed_file: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_rag() -> SimpleLegalRAG:
    """Dependency: the RAG engine, or HTTP 503 if it failed to start."""
    if legal_rag is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized - check the server log")
    return legal_rag


def get_session_id(request: Request) -> str:
    """Session id from the browser cookie, or a new random one."""
    return request.cookies.get(SESSION_COOKIE_NAME) or str(uuid.uuid4())


def set_session_cookie(response: Response, session_id: str) -> None:
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_id,
        max_age=60 * 60 * 24 * SESSION_EXPIRY_DAYS,
        httponly=True,   # not readable by page scripts
        samesite="lax",
    )


def safe_document_path(rag: SimpleLegalRAG, filename: str) -> str:
    """Turn a user-supplied file name into a path inside the documents folder.

    Rejects names like "../api.py" so nobody can read, overwrite or delete
    files outside `legal_documents/`.
    """
    name = (filename or "").replace("\\", "/").split("/")[-1].strip()
    if not name or name.startswith(".") or name != filename.strip():
        raise HTTPException(status_code=400, detail=f"Invalid file name: {filename!r}")

    docs_dir = os.path.abspath(rag.docs_dir)
    path = os.path.abspath(os.path.join(docs_dir, name))
    if os.path.dirname(path) != docs_dir:
        raise HTTPException(status_code=400, detail=f"Invalid file name: {filename!r}")
    return path


# ---------------------------------------------------------------------------
# Page and health check
# ---------------------------------------------------------------------------

@app.get("/")
def get_homepage():
    return FileResponse(os.path.join(BASE_DIR, "index.html"), media_type="text/html")


@app.get("/api/test")
def test_endpoint(response: Response):
    """Health check (used by Docker).

    Returns HTTP 503 when the RAG engine failed to start, so Docker marks the
    container "unhealthy" instead of hiding the failure behind "healthy".
    """
    if legal_rag is None:
        response.status_code = 503
    return {
        "status": "API is working" if legal_rag else "RAG system failed to start - check the logs",
        "timestamp": datetime.now().isoformat(),
        "rag_initialized": legal_rag is not None,
    }


# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------

@app.post("/api/query", response_model=QueryResponse)
def query_rag_api(body: QueryRequest, request: Request, response: Response, rag: SimpleLegalRAG = Depends(get_rag)):
    """Answer a question about the uploaded documents."""
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty")

    session_id = get_session_id(request)
    try:
        result = rag.query(question, session_id=session_id)
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(status_code=502, detail=f"The language model call failed: {e}")

    logger.info("Answered question (session %s..., search query: %r, %d sources)",
                session_id[:8], result["search_query"], len(result["sources"]))
    set_session_cookie(response, session_id)
    return QueryResponse(
        question=result["question"],
        answer=result["answer"],
        sources=result["sources"],
        session_id=session_id,
    )


@app.post("/query", response_model=QueryResponse)
def query_rag_legacy(body: QueryRequest, request: Request, response: Response, rag: SimpleLegalRAG = Depends(get_rag)):
    """Old URL kept for backward compatibility; same as /api/query."""
    return query_rag_api(body, request, response, rag)


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

@app.post("/api/upload", response_model=DocumentResponse)
def upload_document(file: UploadFile = File(...), rag: SimpleLegalRAG = Depends(get_rag)):
    """Save an uploaded file into the documents folder and index it.

    Images are converted to text with OCR. A file with the same name must be
    deleted first; uploads never overwrite an existing document silently.
    """
    final_path = safe_document_path(rag, file.filename)
    file_name = os.path.basename(final_path)
    extension = os.path.splitext(file_name)[1].lower()

    supported = rag.get_supported_file_types()
    if extension not in supported:
        if extension in rag.image_extensions:
            raise HTTPException(status_code=400, detail=f"Cannot process images: {rag.ocr_message}")
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{extension}'. Please upload one of: {', '.join(supported)}")

    # Refuse to overwrite an existing document (or an image's existing text file).
    if os.path.exists(final_path):
        raise HTTPException(status_code=409, detail=f"'{file_name}' already exists. Delete it first to replace it.")
    if rag.is_image_file(file_name):
        text_name = f"{os.path.splitext(file_name)[0]}_extracted.txt"
        if os.path.exists(os.path.join(rag.docs_dir, text_name)):
            raise HTTPException(status_code=409, detail=f"'{text_name}' already exists. Delete it first to replace it.")

    # Write to a hidden temporary name first, then rename. The document list
    # skips hidden files, so a half-written upload never shows up.
    temp_path = os.path.join(rag.docs_dir, f".upload-{uuid.uuid4().hex}{extension}")
    try:
        with open(temp_path, "wb") as out:
            shutil.copyfileobj(file.file, out)
        os.replace(temp_path, final_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    try:
        result = rag.add_document(final_path)
    except Exception as e:
        # Processing failed: remove the saved file (and any chunks) so a
        # broken document doesn't linger in the list.
        logger.exception("Failed to process %s", file_name)
        if os.path.exists(final_path):
            os.remove(final_path)
        rag.remove_from_index(file_name)
        raise HTTPException(status_code=422, detail=f"Could not process '{file_name}': {e}")

    return DocumentResponse(
        filename=file_name,
        status="success",
        message=result["message"],
        converted_from_image=result["converted_from_image"],
        processed_file=result["processed_file"],
    )


@app.get("/api/documents", response_model=List[str])
def list_documents(rag: SimpleLegalRAG = Depends(get_rag)):
    """File names of all documents in the documents folder."""
    return rag.list_document_files()


@app.delete("/api/documents/{filename}")
def delete_document(filename: str, rag: SimpleLegalRAG = Depends(get_rag)):
    """Delete a document from disk and from the vector database."""
    path = safe_document_path(rag, filename)
    file_name = os.path.basename(path)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"Document '{file_name}' not found")

    removed_chunks = rag.remove_document(file_name)
    return {
        "filename": file_name,
        "status": "success",
        "message": f"Document '{file_name}' deleted ({removed_chunks} chunks removed from the index)",
        "chunks_removed": removed_chunks,
    }


@app.post("/api/documents/cleanup")
def cleanup_documents(rag: SimpleLegalRAG = Depends(get_rag)):
    """Re-sync the index with the folder: index new/changed files and drop
    chunks of files that no longer exist."""
    summary = rag.sync_index()
    return {"message": "Cleanup completed", "cleaned": len(summary["removed"]), **summary}


@app.post("/api/force-rebuild")
def force_rebuild(rag: SimpleLegalRAG = Depends(get_rag)):
    """Empty the vector database and re-index every document."""
    summary = rag.rebuild_index()
    return {
        "success": not summary["failed"],
        "message": f"Vector database rebuilt: {len(summary['added'])} documents indexed, {len(summary['failed'])} failed",
        **summary,
    }


@app.post("/api/reset")
def reset_system(rag: SimpleLegalRAG = Depends(get_rag)):
    """Forget all conversations and rebuild the index. Documents are kept."""
    summary = rag.reset()
    return {"status": "success", "message": "Conversations cleared and index rebuilt", **summary}


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

@app.get("/api/session/current")
def get_current_session(request: Request, response: Response, rag: SimpleLegalRAG = Depends(get_rag)):
    """Info about this browser's conversation. Also sets the session cookie,
    so the page and the server agree on the session from the start."""
    session_id = get_session_id(request)
    set_session_cookie(response, session_id)
    count = rag.session_message_count(session_id)
    return {"session_id": session_id, "exists": count > 0, "message_count": count, "created": count > 0}


@app.post("/api/session/clear")
def clear_session(request: Request, rag: SimpleLegalRAG = Depends(get_rag)):
    """Forget this browser's conversation history."""
    if rag.clear_session(get_session_id(request)):
        return {"status": "success", "message": "Session conversation cleared"}
    return {"status": "success", "message": "No existing session to clear"}


# ---------------------------------------------------------------------------
# Status / debugging
# ---------------------------------------------------------------------------

@app.get("/api/system/status")
def get_system_status():
    if legal_rag is None:
        return {"system_initialized": False, "error": "RAG system not initialized - check the server log", "ocr_available": False}
    return {"system_initialized": True, **legal_rag.get_system_status()}


@app.get("/api/ocr-status")
def get_ocr_status():
    if legal_rag is None:
        return {"ocr_available": False, "supported_image_formats": [], "message": "RAG system not initialized"}
    return {
        "ocr_available": legal_rag.ocr_available,
        "supported_image_formats": legal_rag.image_extensions if legal_rag.ocr_available else [],
        "message": legal_rag.ocr_message,
    }


@app.get("/api/debug/system")
def debug_system():
    """Everything in /api/system/status plus the timestamp and folder paths."""
    info = {"timestamp": datetime.now().isoformat(), "rag_initialized": legal_rag is not None}
    if legal_rag is not None:
        info.update(legal_rag.get_system_status())
        info["docs_dir"] = os.path.abspath(legal_rag.docs_dir)
        info["persist_dir"] = os.path.abspath(legal_rag.persist_dir)
    return info


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000)
