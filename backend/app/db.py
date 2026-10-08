"""Tiny SQLite store for document metadata + ingestion status.
Swap for Postgres (SQLAlchemy) when you deploy."""
import sqlite3
import time
from .config import DATA_DIR

DB_PATH = DATA_DIR / "app.db"
FIELDS = {"status", "error", "pages", "progress", "n_chunks"}


def _conn():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with _conn() as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS documents(
            id TEXT PRIMARY KEY, user_id TEXT, filename TEXT,
            status TEXT, error TEXT, pages INTEGER DEFAULT 0,
            progress REAL DEFAULT 0, n_chunks INTEGER DEFAULT 0,
            created_at REAL)"""
        )


def create_doc(doc_id, user_id, filename):
    with _conn() as c:
        c.execute(
            "INSERT INTO documents(id,user_id,filename,status,created_at) VALUES(?,?,?,?,?)",
            (doc_id, user_id, filename, "queued", time.time()),
        )


def update_doc(doc_id, **fields):
    assert set(fields) <= FIELDS
    sets = ",".join(f"{k}=?" for k in fields)
    with _conn() as c:
        c.execute(f"UPDATE documents SET {sets} WHERE id=?", (*fields.values(), doc_id))


def get_doc(doc_id, user_id):
    with _conn() as c:
        r = c.execute(
            "SELECT * FROM documents WHERE id=? AND user_id=?", (doc_id, user_id)
        ).fetchone()
    return dict(r) if r else None


def list_docs(user_id):
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM documents WHERE user_id=? ORDER BY created_at DESC", (user_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def delete_doc(doc_id):
    with _conn() as c:
        c.execute("DELETE FROM documents WHERE id=?", (doc_id,))
