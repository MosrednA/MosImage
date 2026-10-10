from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

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

MIGRATIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    name TEXT PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

Migration = tuple[str, Callable[[sqlite3.Connection], object]]


def ensure_storage() -> None:
    for directory in [
        settings.data_dir,
        settings.input_dir,
        settings.mask_dir,
        settings.output_dir,
        settings.document_dir,
        settings.revision_dir,
        settings.reference_dir,
        settings.layer_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
    with connect() as db:
        # WAL lets the UI read history while a job writes its result. The mode
        # is stored in the database file, so this only changes it once.
        db.execute("PRAGMA journal_mode=WAL")
        apply_migrations(db, GENERATION_MIGRATIONS)


def apply_migrations(db: sqlite3.Connection, migrations: Sequence[Migration]) -> None:
    """Runs each named migration once, in order, and records it.

    Steps must stay idempotent: databases created before migrations were
    recorded already contain their effects without a record of them.
    """
    db.execute(MIGRATIONS_SCHEMA)
    applied = {row["name"] for row in db.execute("SELECT name FROM schema_migrations")}
    for name, step in migrations:
        if name in applied:
            continue
        step(db)
        db.execute("INSERT INTO schema_migrations (name) VALUES (?)", (name,))
        db.commit()


def add_column_if_missing(db: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    columns = {row["name"] for row in db.execute(f"PRAGMA table_info({table})")}
    if column not in columns:
        db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _add_generation_edit_metadata(db: sqlite3.Connection) -> None:
    for column, definition in (("edit_operation", "TEXT"), ("blend_feather", "REAL"), ("context_padding", "INTEGER"), ("scene_context", "TEXT"), ("reference_image_paths", "TEXT")):
        add_column_if_missing(db, "generations", column, definition)


def _add_quality_options(db: sqlite3.Connection) -> None:
    for column in ("consistency_lora", "detail_crop", "specialized_outpaint"):
        add_column_if_missing(db, "generations", column, "INTEGER NOT NULL DEFAULT 0")


GENERATION_MIGRATIONS: list[Migration] = [
    ("generations.0001_create", lambda db: db.execute(SCHEMA)),
    ("generations.0002_runtime_mode", lambda db: add_column_if_missing(db, "generations", "runtime_mode", "TEXT NOT NULL DEFAULT 'balanced'")),
    ("generations.0003_edit_metadata", _add_generation_edit_metadata),
    ("generations.0004_quality_options", _add_quality_options),
]


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


def absolute_path(relative: str) -> Path:
    """A stored relative path inside the data directory; refuses anything outside it."""
    path = (settings.data_dir / relative).resolve()
    path.relative_to(settings.data_dir.resolve())
    return path


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


def delete_generation(generation_id: int) -> None:
    """Removes a generation and the files only it uses.

    Proposals made together share one input and mask; those stay until the
    last proposal that uses them is gone. Documents copy what they keep, so
    deleting a generation never breaks a document.
    """
    with connect() as db:
        row = db.execute("SELECT * FROM generations WHERE id = ?", (generation_id,)).fetchone()
        if row is None:
            raise LookupError("Generation not found")
        db.execute("DELETE FROM generations WHERE id = ?", (generation_id,))
        orphaned = [row["output_image_path"]]
        for column in ("input_image_path", "mask_image_path"):
            relative = row[column]
            if relative and db.execute(f"SELECT 1 FROM generations WHERE {column} = ? LIMIT 1", (relative,)).fetchone() is None:
                orphaned.append(relative)
        db.commit()
    for relative in orphaned:
        absolute_path(relative).unlink(missing_ok=True)


def row_to_record(row: sqlite3.Row) -> dict[str, Any]:
    record = dict(row)
    record["output_image_url"] = image_url(record["output_image_path"])
    record["reference_image_paths"] = json.loads(record.get("reference_image_paths") or "[]")
    for column in ("consistency_lora", "detail_crop", "specialized_outpaint"):
        record[column] = bool(record[column])
    return record
