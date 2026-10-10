"""Concept images used as appearance references.

Each belongs to a document. Loose ones, without a document, wait on the start
screen and move into the next new document, so every project keeps its own.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import re
import sqlite3
from sqlite3 import Row
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from . import storage
from .config import settings


REFERENCE_SCHEMA = """
CREATE TABLE IF NOT EXISTS document_references (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    image_path TEXT NOT NULL,
    subject_path TEXT,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY(document_id) REFERENCES documents(id)
);

CREATE INDEX IF NOT EXISTS document_references_document_idx ON document_references(document_id, id);
"""

MAX_REFERENCES = 24
MAX_PIXELS = 16_777_216
# Neutral studio grey: concept images on plain backgrounds worked best in trials.
SUBJECT_BACKGROUND = (232, 232, 232)


def _allow_loose_references(db: sqlite3.Connection) -> None:
    """SQLite cannot drop NOT NULL in place, so the table is rebuilt once."""
    columns = {row["name"]: row for row in db.execute("PRAGMA table_info(document_references)")}
    if not columns["document_id"]["notnull"]:
        return
    db.executescript("""
        CREATE TABLE document_references_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER,
            name TEXT NOT NULL,
            image_path TEXT NOT NULL,
            subject_path TEXT,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY(document_id) REFERENCES documents(id)
        );
        INSERT INTO document_references_new SELECT id, document_id, name, image_path, subject_path, width, height, created_at FROM document_references;
        DROP TABLE document_references;
        ALTER TABLE document_references_new RENAME TO document_references;
        CREATE INDEX IF NOT EXISTS document_references_document_idx ON document_references(document_id, id);
    """)


# Names cameras, phones, chat apps and generators give files; nobody types these.
MACHINE_PREFIXES = {
    "img", "image", "dsc", "dscn", "dscf", "pxl", "mvimg", "vid", "photo", "picture", "screenshot",
    "schermafbeelding", "scherm", "whatsapp", "signal", "telegram", "chatgpt", "gemini", "dalle", "download", "unnamed",
}
NAME_TOKEN = re.compile(r"[\s_.,()\[\]-]+")


def _is_machine_name(stem: str) -> bool:
    """True for file names such as IMG_1234, PXL_20261009_142211 or T5S8DAP28-7b1f1113fc48."""
    tokens = [token for token in NAME_TOKEN.split(stem.lower()) if token]
    if not tokens:
        return True
    if tokens[0] in MACHINE_PREFIXES and any(character.isdigit() for character in stem):
        return True
    # Hashes and IDs mix letters and digits; timestamps are long digit runs.
    return any(
        (len(token) >= 6 and any(c.isdigit() for c in token) and any(c.isalpha() for c in token)) or re.search(r"\d{5}", token)
        for token in tokens
    )


def _number_machine_names(db: sqlite3.Connection) -> None:
    """Gives references named after code-like files a numbered name, per document."""
    rows = db.execute("SELECT id, document_id, name FROM document_references ORDER BY document_id, id").fetchall()
    positions: dict[int | None, int] = {}
    for row in rows:
        position = positions[row["document_id"]] = positions.get(row["document_id"], 0) + 1
        if _is_machine_name(row["name"]):
            db.execute("UPDATE document_references SET name = ? WHERE id = ?", (f"Reference {position}", row["id"]))


REFERENCE_MIGRATIONS: list[storage.Migration] = [
    ("references.0001_create", lambda db: db.executescript(REFERENCE_SCHEMA)),
    ("references.0002_loose", _allow_loose_references),
    ("references.0003_machine_names", _number_machine_names),
]


def ensure_reference_storage() -> None:
    settings.reference_dir.mkdir(parents=True, exist_ok=True)
    with storage.connect() as db:
        storage.apply_migrations(db, REFERENCE_MIGRATIONS)


def list_references(document_id: int | None) -> list[dict[str, Any]]:
    """A document's references, or the loose ones for ``None``."""
    with storage.connect() as db:
        rows = db.execute(
            "SELECT * FROM document_references WHERE document_id IS ? ORDER BY id",
            (document_id,),
        ).fetchall()
    return [_record(row) for row in rows]


def get_reference(reference_id: int) -> dict[str, Any] | None:
    with storage.connect() as db:
        row = db.execute("SELECT * FROM document_references WHERE id = ?", (reference_id,)).fetchone()
    return _record(row) if row else None


def create_reference(document_id: int | None, content: bytes, filename: str | None) -> dict[str, Any]:
    with storage.connect() as db:
        count = _count(db, document_id)
    if count >= MAX_REFERENCES:
        holder = "A document" if document_id is not None else "The start screen"
        raise ValueError(f"{holder} can hold up to {MAX_REFERENCES} references")
    path = settings.reference_dir / f"reference-{uuid4().hex}.png"
    try:
        # Normalize to PNG: browsers and the model see the same pixels, and
        # EXIF rotation from phone photos is applied once.
        with Image.open(BytesIO(content)) as uploaded:
            if uploaded.width * uploaded.height > MAX_PIXELS:
                raise ValueError("Reference images can be up to 16 megapixels")
            image = ImageOps.exif_transpose(uploaded).convert("RGBA" if "A" in uploaded.getbands() else "RGB")
        image.save(path)
        with storage.connect() as db:
            cursor = db.execute(
                "INSERT INTO document_references (document_id, name, image_path, width, height) VALUES (?, ?, ?, ?, ?)",
                (document_id, _clean_name(filename, count + 1), storage.to_relative(path), image.width, image.height),
            )
            db.commit()
            reference_id = int(cursor.lastrowid)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return get_reference(reference_id) or {}


def adopt_loose_references(document_id: int) -> list[dict[str, Any]]:
    """Moves the start screen's references into a new document, up to its limit; ids stay the same."""
    with storage.connect() as db:
        room = MAX_REFERENCES - _count(db, document_id)
        ids = [row["id"] for row in db.execute(
            "SELECT id FROM document_references WHERE document_id IS NULL ORDER BY id LIMIT ?", (max(room, 0),)
        )]
        db.executemany("UPDATE document_references SET document_id = ? WHERE id = ?", [(document_id, item) for item in ids])
        db.commit()
    return list_references(document_id)


def set_subject(reference_id: int, subject: Image.Image) -> dict[str, Any]:
    reference = get_reference(reference_id)
    if reference is None:
        raise LookupError("Reference not found")
    path = settings.reference_dir / f"subject-{uuid4().hex}.png"
    subject.save(path)
    previous = reference["subject_path"]
    with storage.connect() as db:
        db.execute("UPDATE document_references SET subject_path = ? WHERE id = ?", (storage.to_relative(path), reference_id))
        db.commit()
    if previous:
        _absolute(previous).unlink(missing_ok=True)
    return get_reference(reference_id) or {}


def extract_subject(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Place the masked subject on a neutral background, cropped with margin.

    The model then copies the subject's appearance without also pulling in
    unrelated people, signs or scenery from the concept image.
    """
    alpha = Image.fromarray(np.where(mask > 0, 255, 0).astype(np.uint8), mode="L")
    box = alpha.getbbox()
    if box is None:
        raise ValueError("The subject mask is empty")
    margin = round(max(box[2] - box[0], box[3] - box[1]) * 0.06)
    box = (max(box[0] - margin, 0), max(box[1] - margin, 0), min(box[2] + margin, image.width), min(box[3] + margin, image.height))
    # A one-pixel soft edge avoids a hard cut-out halo.
    alpha = alpha.filter(ImageFilter.GaussianBlur(1))
    subject = Image.new("RGB", image.size, SUBJECT_BACKGROUND)
    subject.paste(image.convert("RGB"), (0, 0), alpha)
    return subject.crop(box)


def clear_subject(reference_id: int) -> dict[str, Any]:
    reference = get_reference(reference_id)
    if reference is None:
        raise LookupError("Reference not found")
    if reference["subject_path"]:
        with storage.connect() as db:
            db.execute("UPDATE document_references SET subject_path = NULL WHERE id = ?", (reference_id,))
            db.commit()
        _absolute(reference["subject_path"]).unlink(missing_ok=True)
    return get_reference(reference_id) or {}


def delete_reference(reference_id: int) -> None:
    reference = get_reference(reference_id)
    if reference is None:
        raise LookupError("Reference not found")
    with storage.connect() as db:
        db.execute("DELETE FROM document_references WHERE id = ?", (reference_id,))
        db.commit()
    for relative in (reference["image_path"], reference["subject_path"]):
        if relative:
            _absolute(relative).unlink(missing_ok=True)


def original_image_path(reference_id: int) -> Path:
    reference = get_reference(reference_id)
    if reference is None:
        raise LookupError("Reference not found")
    return _absolute(reference["image_path"])


def model_image_path(reference_id: int) -> Path:
    """The image the model receives: the cut-out subject when available."""
    reference = get_reference(reference_id)
    if reference is None:
        raise LookupError("Reference not found")
    return _absolute(reference["subject_path"] or reference["image_path"])


def _count(db: sqlite3.Connection, document_id: int | None) -> int:
    if document_id is not None and db.execute("SELECT 1 FROM documents WHERE id = ?", (document_id,)).fetchone() is None:
        raise LookupError("Document not found")
    return db.execute("SELECT COUNT(*) FROM document_references WHERE document_id IS ?", (document_id,)).fetchone()[0]


def _record(row: Row) -> dict[str, Any]:
    record = dict(row)
    record["image_url"] = storage.image_url(record["image_path"])
    record["subject_url"] = storage.image_url(record["subject_path"]) if record["subject_path"] else None
    return record


def _absolute(relative: str) -> Path:
    path = (settings.data_dir / relative).resolve()
    path.relative_to(settings.data_dir.resolve())
    return path


def _clean_name(filename: str | None, position: int) -> str:
    stem = Path(filename or "").stem.strip()
    if not stem or stem.lower() in {"image", "blob", "pasted image"} or _is_machine_name(stem):
        return f"Reference {position}"
    return stem[:80]
