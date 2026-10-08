"""PDF -> LangChain Documents (text, tables, visuals) -> Pinecone.
Runs as a background task; progress/status are written to the DB."""
import traceback
from pathlib import Path

import pymupdf as fitz
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import db, vectorstore
from .config import MAX_PAGES, doc_dir
from .vision import summarize_visual

splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
MIN_IMG_PX = 120  # skip icons/logos


def extract_documents(pdf_path: Path, doc_id, user_id, filename, images_dir, on_progress):
    pdf = fitz.open(str(pdf_path))
    total = min(len(pdf), MAX_PAGES)
    docs, seen = [], set()

    def meta(page, modality, **extra):
        return {"page": page, "modality": modality, "doc_id": doc_id,
                "user_id": user_id, "source": filename, **extra}

    for i in range(total):
        page, n = pdf[i], i + 1
        text = page.get_text("text").strip()
        has_content = bool(text)

        # 1. TEXT (chunked so long pages embed well)
        for chunk in splitter.split_text(text) if text else []:
            docs.append(Document(page_content=chunk, metadata=meta(n, "text")))

        # 2. TABLES
        try:
            for t_no, table in enumerate(page.find_tables().tables, 1):
                df = table.to_pandas()
                if not df.empty:
                    has_content = True
                    docs.append(Document(
                        page_content=df.to_markdown(index=False),
                        metadata=meta(n, "table", table_number=t_no)))
        except Exception as e:
            print(f"[p{n}] table warning: {e}")

        # 3. EMBEDDED IMAGES / CHARTS / DIAGRAMS
        for img_no, info in enumerate(page.get_images(full=True), 1):
            xref = info[0]
            if xref in seen:
                continue
            seen.add(xref)
            data = pdf.extract_image(xref)
            if min(data.get("width", 0), data.get("height", 0)) < MIN_IMG_PX:
                continue
            path = images_dir / f"page_{n}_image_{img_no}.{data['ext']}"
            path.write_bytes(data["image"])
            try:
                has_content = True
                docs.append(Document(
                    page_content=summarize_visual(path, n),
                    metadata=meta(n, "visual", image=path.name)))
            except Exception as e:
                print(f"[p{n}] vision warning: {e}")

        # 4. SCANNED-PAGE FALLBACK (OCR via the vision model)
        if not has_content:
            path = images_dir / f"page_{n}_scan.png"
            page.get_pixmap(dpi=150).save(str(path))
            try:
                docs.append(Document(
                    page_content=summarize_visual(path, n),
                    metadata=meta(n, "visual", image=path.name)))
            except Exception as e:
                print(f"[p{n}] scan OCR warning: {e}")

        on_progress((n / total) * 0.9)  # last 10% = embedding/upsert

    pdf.close()
    return docs, total


def process_document(doc_id: str, user_id: str, pdf_path: Path, filename: str):
    try:
        db.update_doc(doc_id, status="processing", progress=0.02)
        images_dir = doc_dir(doc_id) / "images"
        docs, pages = extract_documents(
            pdf_path, doc_id, user_id, filename, images_dir,
            lambda p: db.update_doc(doc_id, progress=p))
        if not docs:
            raise ValueError("No readable content found in this PDF.")
        ids = [f"{doc_id}:{i}" for i in range(len(docs))]
        vectorstore.add_docs(user_id, docs, ids)
        db.update_doc(doc_id, status="ready", progress=1.0, pages=pages, n_chunks=len(docs))
    except Exception as e:
        traceback.print_exc()
        db.update_doc(doc_id, status="failed", error=str(e)[:300])
