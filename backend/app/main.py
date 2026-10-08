import re
import uuid

import pymupdf as fitz
from fastapi import BackgroundTasks, Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from . import db, rag, vectorstore
from .config import MAX_UPLOAD_MB, MODELS, doc_dir, get_model, require_secrets
from .ingestion import process_document

app = FastAPI(title="Report Chat API", version="1.0")


@app.on_event("startup")
def startup():
    require_secrets()
    db.init_db()
    vectorstore.ensure_index()


def current_user(x_user_id: str = Header(...)) -> str:
    """MVP auth: a workspace id header. Replace with JWT (Clerk/Supabase) in prod."""
    if not re.fullmatch(r"[a-zA-Z0-9_-]{3,64}", x_user_id):
        raise HTTPException(400, "X-User-Id must be 3-64 chars: letters, digits, - or _")
    return x_user_id


def own_doc(doc_id: str, user: str):
    d = db.get_doc(doc_id, user)
    if not d:
        raise HTTPException(404, "Document not found")
    return d


class ChatRequest(BaseModel):
    question: str
    doc_ids: list[str]
    history: list[dict] = []
    model: str | None = None


@app.get("/models")
def list_models():
    return [{"key": m["key"], "label": m["label"], "description": m.get("description", ""),
             "vision": m.get("vision", False)} for m in MODELS]


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/documents", status_code=202)
async def upload(bg: BackgroundTasks, file: UploadFile = File(...), user: str = Depends(current_user)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File larger than {MAX_UPLOAD_MB} MB")
    if not data.startswith(b"%PDF"):
        raise HTTPException(400, "File is not a valid PDF")

    doc_id = uuid.uuid4().hex[:12]
    pdf_path = doc_dir(doc_id) / "original.pdf"
    pdf_path.write_bytes(data)
    db.create_doc(doc_id, user, file.filename)
    bg.add_task(process_document, doc_id, user, pdf_path, file.filename)
    return {"id": doc_id, "status": "queued"}


@app.get("/documents")
def list_documents(user: str = Depends(current_user)):
    return db.list_docs(user)


@app.get("/documents/{doc_id}")
def get_document(doc_id: str, user: str = Depends(current_user)):
    return own_doc(doc_id, user)


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str, user: str = Depends(current_user)):
    d = own_doc(doc_id, user)
    if d["n_chunks"]:
        vectorstore.delete_ids(user, [f"{doc_id}:{i}" for i in range(d["n_chunks"])])
    import shutil
    shutil.rmtree(doc_dir(doc_id), ignore_errors=True)
    db.delete_doc(doc_id)
    return {"deleted": doc_id}


@app.get("/documents/{doc_id}/pages/{page}")
def render_page(doc_id: str, page: int, user: str = Depends(current_user)):
    """Renders a PDF page as PNG so the UI can show the cited page."""
    own_doc(doc_id, user)
    pdf = fitz.open(str(doc_dir(doc_id) / "original.pdf"))
    if not 1 <= page <= len(pdf):
        raise HTTPException(404, "Page out of range")
    png = pdf[page - 1].get_pixmap(dpi=110).tobytes("png")
    pdf.close()
    return Response(png, media_type="image/png")


@app.post("/chat")
def chat(req: ChatRequest, user: str = Depends(current_user)):
    if not req.question.strip():
        raise HTTPException(400, "Empty question")
    if len(req.question) > 2000:
        raise HTTPException(400, "Question too long")
    try:
        get_model(req.model)
    except KeyError:
        raise HTTPException(400, f"Unknown model '{req.model}'")
    return rag.ask(user, req.question.strip(), req.doc_ids, req.history, req.model)
