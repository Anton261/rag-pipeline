"""Async SQLite repository using aiosqlite."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiosqlite


# ---------------------------------------------------------------------------
# Data records
# ---------------------------------------------------------------------------

@dataclass
class DocumentRecord:
    id: str
    filename: str
    page_count: int
    chunk_count: int
    ingested_at: str  # ISO-8601


@dataclass
class ChunkRecord:
    id: str
    doc_id: str
    chunk_index: int
    page_number: int
    text: str
    token_count: int
    npy_path: str


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------

class Repository:
    def __init__(self, db_path: Path) -> None:
        self._db_path = str(db_path)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    async def init_db(self) -> None:
        schema_path = Path(__file__).parent / "schema.sql"
        schema = schema_path.read_text(encoding="utf-8")
        async with aiosqlite.connect(self._db_path) as db:
            await db.executescript(schema)
            await db.commit()

    # ------------------------------------------------------------------
    # Documents
    # ------------------------------------------------------------------

    async def insert_document(self, doc: DocumentRecord) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT INTO documents (id, filename, page_count, chunk_count, ingested_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (doc.id, doc.filename, doc.page_count, doc.chunk_count, doc.ingested_at),
            )
            await db.commit()

    async def list_documents(self) -> list[DocumentRecord]:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT id, filename, page_count, chunk_count, ingested_at "
                "FROM documents ORDER BY ingested_at DESC"
            ) as cursor:
                rows = await cursor.fetchall()
        return [
            DocumentRecord(
                id=row["id"],
                filename=row["filename"],
                page_count=row["page_count"],
                chunk_count=row["chunk_count"],
                ingested_at=row["ingested_at"],
            )
            for row in rows
        ]

    async def document_exists(self, filename: str) -> bool:
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT 1 FROM documents WHERE filename = ?", (filename,)
            ) as cursor:
                return await cursor.fetchone() is not None

    async def delete_document(self, doc_id: str) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            await db.commit()

    # ------------------------------------------------------------------
    # Chunks
    # ------------------------------------------------------------------

    async def insert_chunks(self, chunks: list[ChunkRecord]) -> None:
        rows = [
            (c.id, c.doc_id, c.chunk_index, c.page_number, c.text, c.token_count, c.npy_path)
            for c in chunks
        ]
        async with aiosqlite.connect(self._db_path) as db:
            await db.executemany(
                "INSERT INTO chunks (id, doc_id, chunk_index, page_number, text, token_count, npy_path) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            await db.commit()

    async def get_all_chunks(self) -> list[ChunkRecord]:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT id, doc_id, chunk_index, page_number, text, token_count, npy_path "
                "FROM chunks"
            ) as cursor:
                rows = await cursor.fetchall()
        return [_row_to_chunk(row) for row in rows]

    async def get_chunks_by_ids(self, ids: list[str]) -> list[ChunkRecord]:
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                f"SELECT id, doc_id, chunk_index, page_number, text, token_count, npy_path "
                f"FROM chunks WHERE id IN ({placeholders})",
                ids,
            ) as cursor:
                rows = await cursor.fetchall()
        # Preserve requested order
        row_map = {row["id"]: _row_to_chunk(row) for row in rows}
        return [row_map[i] for i in ids if i in row_map]

    async def get_chunk_count(self) -> int:
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute("SELECT COUNT(*) FROM chunks") as cursor:
                row = await cursor.fetchone()
        return row[0] if row else 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row_to_chunk(row: Any) -> ChunkRecord:
    return ChunkRecord(
        id=row["id"],
        doc_id=row["doc_id"],
        chunk_index=row["chunk_index"],
        page_number=row["page_number"],
        text=row["text"],
        token_count=row["token_count"],
        npy_path=row["npy_path"],
    )
