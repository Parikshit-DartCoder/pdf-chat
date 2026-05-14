"""Per-session conversation memory backed by SQLite. Stores message turns and
a running summary so prompts stay bounded as conversations grow."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlmodel import Field, Session, SQLModel, create_engine, select

from ..config.settings import get_settings


class Turn(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(index=True)
    role: str  # "user" | "assistant"
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Summary(SQLModel, table=True):
    session_id: str = Field(primary_key=True)
    summary: str
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class SessionDocument(SQLModel, table=True):
    """A document the user added (uploaded or referenced) during a session.
    Used to scope retrieval to 'this session's documents' when the user says
    'this doc'."""
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(index=True)
    source_path: str
    display_name: str
    added_at: datetime = Field(default_factory=datetime.utcnow)


def _engine():
    s = get_settings()
    return create_engine(f"sqlite:///{s.memory_db_path}", echo=False)


def init_memory() -> None:
    SQLModel.metadata.create_all(_engine())


def append_turn(session_id: str, role: str, content: str) -> None:
    with Session(_engine()) as ses:
        ses.add(Turn(session_id=session_id, role=role, content=content))
        ses.commit()


def recent_turns(session_id: str, limit: int = 8) -> list[Turn]:
    with Session(_engine()) as ses:
        rows = ses.exec(
            select(Turn)
            .where(Turn.session_id == session_id)
            .order_by(Turn.id.desc())
            .limit(limit)
        ).all()
    return list(reversed(rows))


def get_summary(session_id: str) -> str | None:
    with Session(_engine()) as ses:
        row = ses.get(Summary, session_id)
        return row.summary if row else None


def set_summary(session_id: str, summary: str) -> None:
    with Session(_engine()) as ses:
        row = ses.get(Summary, session_id)
        if row:
            row.summary = summary
            row.updated_at = datetime.utcnow()
        else:
            row = Summary(session_id=session_id, summary=summary)
        ses.add(row)
        ses.commit()


def add_session_document(session_id: str, source_path: str, display_name: str) -> None:
    """Register that this session has access to a document. Idempotent."""
    with Session(_engine()) as ses:
        existing = ses.exec(
            select(SessionDocument)
            .where(SessionDocument.session_id == session_id)
            .where(SessionDocument.source_path == source_path)
        ).first()
        if existing:
            return
        ses.add(SessionDocument(
            session_id=session_id, source_path=source_path, display_name=display_name,
        ))
        ses.commit()


def list_session_documents(session_id: str) -> list[SessionDocument]:
    with Session(_engine()) as ses:
        return list(ses.exec(
            select(SessionDocument)
            .where(SessionDocument.session_id == session_id)
            .order_by(SessionDocument.id.asc())
        ).all())


def list_corpus_documents() -> list[str]:
    """Distinct source_paths ever ingested (any session, including CLI bulk).
    Returns paths sorted alphabetically. Reads from Qdrant via scroll."""
    from qdrant_client import QdrantClient
    s = get_settings()
    client = QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key, timeout=30)
    seen: set[str] = set()
    next_page = None
    while True:
        points, next_page = client.scroll(
            collection_name=s.qdrant_collection,
            limit=256,
            with_payload=["source_path"],
            with_vectors=False,
            offset=next_page,
        )
        for p in points:
            sp = (p.payload or {}).get("source_path")
            if sp:
                seen.add(sp)
        if next_page is None:
            break
    return sorted(seen)
