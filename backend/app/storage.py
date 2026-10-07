from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .config import settings


SCHEMA = """
CREATE TABLE IF NOT EXISTS generations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task TEXT NOT NULL,
    prompt TEXT NOT NULL,
    negative_prompt TEXT,
    model_name TEXT NOT NULL,
    runtime_mode TEXT NOT NULL DEFAULT 'balanced',
    seed INTEGER NOT NULL,
    steps INTEGER NOT NULL,
    guidance REAL NOT NULL,
    width INTEGER,
    height INTEGER,
    strength REAL,
    edit_operation TEXT,
    blend_feather REAL,
    context_padding INTEGER,
    scene_context TEXT,
    input_image_path TEXT,
    mask_image_path TEXT,
    output_image_path TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def ensure_storage() -> None:
    for directory in [
        settings.data_dir,
        settings.input_dir,
        settings.mask_dir,
        settings.output_dir,
        settings.document_dir,
        settings.revision_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
    with connect() as db:
        db.execute(SCHEMA)
        _add_column_if_missing(db, "generations", "runtime_mode", "TEXT NOT NULL DEFAULT 'balanced'")
        for column, definition in (("edit_operation", "TEXT"), ("blend_feather", "REAL"), ("context_padding", "INTEGER"), ("scene_context", "TEXT")):
            _add_column_if_missing(db, "generations", column, definition)
        db.commit()


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    db = sqlite3.connect(settings.db_path)
    db.row_factory = sqlite3.Row
    try:
        yield db
    finally:
        db.close()


def to_relative(path: Path | None) -> str | None:
    if path is None:
        return None
    return path.resolve().relative_to(settings.data_dir.resolve()).as_posix()


def image_url(relative_path: str) -> str:
    return f"/files/{relative_path}"


def create_generation(values: dict[str, Any]) -> int:
    return create_generations([values])[0]


def create_generations(records: list[dict[str, Any]]) -> list[int]:
    if not records:
        return []
    generation_ids: list[int] = []
    with connect() as db:
        for values in records:
            columns = ", ".join(values.keys())
            placeholders = ", ".join(f":{key}" for key in values.keys())
            cursor = db.execute(
                f"INSERT INTO generations ({columns}) VALUES ({placeholders})",
                values,
            )
            generation_ids.append(int(cursor.lastrowid))
        db.commit()
    return generation_ids


def list_generations(limit: int = 50) -> list[dict[str, Any]]:
    with connect() as db:
        rows = db.execute(
            "SELECT * FROM generations ORDER BY datetime(created_at) DESC, id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [row_to_record(row) for row in rows]


def get_generation(generation_id: int) -> dict[str, Any] | None:
    with connect() as db:
        row = db.execute("SELECT * FROM generations WHERE id = ?", (generation_id,)).fetchone()
    return row_to_record(row) if row else None


def row_to_record(row: sqlite3.Row) -> dict[str, Any]:
    record = dict(row)
    record["output_image_url"] = image_url(record["output_image_path"])
    return record


def _add_column_if_missing(db: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    columns = {row["name"] for row in db.execute(f"PRAGMA table_info({table})")}
    if column not in columns:
        db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
