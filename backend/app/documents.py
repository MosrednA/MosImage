from __future__ import annotations

import json
import shutil
from pathlib import Path
from sqlite3 import Connection, Row
from typing import Any
from uuid import uuid4

from PIL import Image, ImageOps

from . import layers as layer_store
from . import storage
from .compositing import FAST_PNG, rebase_expansion
from .config import settings


DOCUMENT_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    canvas_width INTEGER NOT NULL,
    canvas_height INTEGER NOT NULL,
    color_profile TEXT NOT NULL DEFAULT 'sRGB',
    current_revision_id INTEGER,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS revisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,
    parent_revision_id INTEGER,
    operation_type TEXT NOT NULL,
    composite_image_path TEXT NOT NULL,
    mask_image_path TEXT,
    operation_parameters TEXT NOT NULL DEFAULT '{}',
    selected_alternative_id INTEGER,
    accepted INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY(document_id) REFERENCES documents(id)
);

CREATE TABLE IF NOT EXISTS alternatives (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    revision_id INTEGER NOT NULL,
    generation_id INTEGER,
    seed INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    output_image_path TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    selected INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY(revision_id) REFERENCES revisions(id)
);

CREATE INDEX IF NOT EXISTS revisions_document_idx ON revisions(document_id, id);
CREATE INDEX IF NOT EXISTS revisions_parent_idx ON revisions(parent_revision_id, accepted, id);
CREATE INDEX IF NOT EXISTS alternatives_revision_idx ON alternatives(revision_id, id);
"""


DOCUMENT_MIGRATIONS: list[storage.Migration] = [
    ("documents.0001_create", lambda db: db.executescript(DOCUMENT_SCHEMA)),
    ("documents.0002_revision_layers", lambda db: storage.add_column_if_missing(db, "revisions", "layers", "TEXT NOT NULL DEFAULT '[]'")),
]


class RevisionConflict(ValueError):
    """The document moved on to another revision while a change was prepared."""


def ensure_document_storage() -> None:
    layer_store.ensure_layer_storage()
    with storage.connect() as db:
        storage.apply_migrations(db, DOCUMENT_MIGRATIONS)


def create_from_upload(name: str, content: bytes, filename: str | None) -> dict[str, Any]:
    path = _save_document_image(content, filename)
    try:
        width, height = _image_size(path)
        with storage.connect() as db:
            cursor = db.execute(
                "INSERT INTO documents (name, canvas_width, canvas_height) VALUES (?, ?, ?)",
                (_clean_name(name, filename), width, height),
            )
            document_id = int(cursor.lastrowid)
            revision_id = _insert_revision(
                db,
                document_id=document_id,
                parent_revision_id=None,
                operation_type="import",
                composite_path=storage.to_relative(path),
                mask_path=None,
                parameters={"filename": filename or "image.png"},
                accepted=True,
            )
            _set_current_revision(db, document_id, revision_id)
            db.commit()
        return get_document(document_id) or _missing_document(document_id)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def import_generation(generation_id: int, name: str | None = None) -> dict[str, Any]:
    generation = storage.get_generation(generation_id)
    if generation is None:
        raise LookupError("Generation not found")
    source = _absolute_path(generation["output_image_path"])
    copied = _copy_revision_asset(source, "import")
    try:
        width, height = _image_size(copied)
        document_name = name or generation["prompt"][:80] or f"Generation {generation_id}"
        with storage.connect() as db:
            cursor = db.execute(
                "INSERT INTO documents (name, canvas_width, canvas_height) VALUES (?, ?, ?)",
                (_clean_name(document_name, None), width, height),
            )
            document_id = int(cursor.lastrowid)
            revision_id = _insert_revision(
                db,
                document_id=document_id,
                parent_revision_id=None,
                operation_type="v1-import",
                composite_path=storage.to_relative(copied),
                mask_path=None,
                parameters={"generation_id": generation_id},
                accepted=True,
            )
            _set_current_revision(db, document_id, revision_id)
            db.commit()
        return get_document(document_id) or _missing_document(document_id)
    except BaseException:
        copied.unlink(missing_ok=True)
        raise


def list_documents(limit: int = 50) -> list[dict[str, Any]]:
    with storage.connect() as db:
        rows = db.execute(
            "SELECT * FROM documents ORDER BY datetime(updated_at) DESC, id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [_document_record(db, row) for row in rows]


def get_document(document_id: int) -> dict[str, Any] | None:
    with storage.connect() as db:
        row = db.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
        return _document_record(db, row) if row else None


def update_document(document_id: int, name: str) -> dict[str, Any]:
    with storage.connect() as db:
        cursor = db.execute(
            "UPDATE documents SET name = ?, updated_at = datetime('now') WHERE id = ?",
            (_clean_name(name, None), document_id),
        )
        if cursor.rowcount == 0:
            raise LookupError("Document not found")
        db.commit()
    return get_document(document_id) or _missing_document(document_id)


def create_revision_from_upload(
    document_id: int,
    content: bytes,
    filename: str | None,
    operation_type: str,
) -> dict[str, Any]:
    document = get_document(document_id)
    if document is None:
        raise LookupError("Document not found")
    path = _save_revision_image(content, filename, "revision")
    try:
        width, height = _image_size(path)
        if (width, height) != (document["canvas_width"], document["canvas_height"]):
            raise ValueError("Revision dimensions must match the document canvas")
        with storage.connect() as db:
            revision_id = _insert_revision(
                db,
                document_id=document_id,
                parent_revision_id=document["current_revision_id"],
                operation_type=operation_type,
                composite_path=storage.to_relative(path),
                mask_path=None,
                parameters={},
                accepted=True,
            )
            _set_current_revision(db, document_id, revision_id)
            db.commit()
        return get_revision(revision_id) or _missing_revision(revision_id)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def create_alternative_proposal(
    document_id: int,
    generation_ids: list[int],
    operation_type: str,
    base_revision_id: int | None = None,
    layer_offset: tuple[int, int] = (0, 0),
) -> dict[str, Any]:
    if len(set(generation_ids)) != len(generation_ids):
        raise ValueError("Alternative generation IDs must be unique")
    document = get_document(document_id)
    if document is None:
        raise LookupError("Document not found")
    if base_revision_id is not None and document["current_revision_id"] != base_revision_id:
        raise RevisionConflict("The document changed while these alternatives were generated")
    generations = [storage.get_generation(item) for item in generation_ids]
    if any(item is None for item in generations):
        raise LookupError("One or more generations were not found")
    resolved = [item for item in generations if item is not None]
    expected_task = operation_type if operation_type in {"inpaint", "outpaint"} else "text-to-image"
    if any(item["task"] != expected_task for item in resolved):
        raise ValueError("Generation task does not match the proposed operation")

    with storage.connect() as db:
        parent = db.execute("SELECT composite_image_path, layers FROM revisions WHERE id = ?", (document["current_revision_id"],)).fetchone()
    parent_layers = layer_store.parse(parent["layers"]) if parent else []
    # Expand continued the picture with its layers, which stay on top of the new background.
    rebase = operation_type == "outpaint" and bool(parent_layers)
    copied_paths: list[Path] = []
    copied_mask: Path | None = None
    proposal_size: tuple[int, int] | None = None
    try:
        for generation in resolved:
            source = _absolute_path(generation["output_image_path"])
            width, height = _image_size(source)
            if operation_type != "outpaint" and (width, height) != (document["canvas_width"], document["canvas_height"]):
                raise ValueError("Alternative dimensions must match the document canvas")
            if operation_type == "outpaint" and (width < document["canvas_width"] or height < document["canvas_height"]):
                raise ValueError("Outpainting cannot shrink the document canvas")
            if proposal_size is not None and proposal_size != (width, height):
                raise ValueError("Alternative dimensions must match each other")
            proposal_size = (width, height)
            copied_paths.append(
                _rebased_expansion(generation, _absolute_path(parent["composite_image_path"]), layer_offset)
                if rebase else _copy_revision_asset(source, "alternative")
            )
        mask_relative = resolved[0].get("mask_image_path")
        if mask_relative:
            copied_mask = _copy_revision_asset(_absolute_path(mask_relative), "mask")

        with storage.connect() as db:
            revision_id = _insert_revision(
                db,
                document_id=document_id,
                parent_revision_id=document["current_revision_id"],
                operation_type=operation_type,
                composite_path=storage.to_relative(copied_paths[0]),
                mask_path=storage.to_relative(copied_mask),
                parameters={"generation_ids": generation_ids},
                accepted=False,
                layers=layer_store.shifted(parent_layers, *layer_offset) if operation_type == "outpaint" else parent_layers,
            )
            for generation, path in zip(resolved, copied_paths, strict=True):
                db.execute(
                    """
                    INSERT INTO alternatives
                        (revision_id, generation_id, seed, model_name, output_image_path, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        revision_id,
                        generation["id"],
                        generation["seed"],
                        generation["model_name"],
                        storage.to_relative(path),
                        json.dumps({"prompt": generation["prompt"], "runtime_mode": generation["runtime_mode"]}),
                    ),
                )
            db.commit()
        return get_revision(revision_id) or _missing_revision(revision_id)
    except BaseException:
        for path in copied_paths:
            path.unlink(missing_ok=True)
        if copied_mask:
            copied_mask.unlink(missing_ok=True)
        raise


def accept_alternative(document_id: int, revision_id: int, alternative_id: int) -> dict[str, Any]:
    with storage.connect() as db:
        revision = db.execute(
            "SELECT * FROM revisions WHERE id = ? AND document_id = ?",
            (revision_id, document_id),
        ).fetchone()
        if revision is None:
            raise LookupError("Revision not found")
        alternative = db.execute(
            "SELECT * FROM alternatives WHERE id = ? AND revision_id = ?",
            (alternative_id, revision_id),
        ).fetchone()
        if alternative is None:
            raise LookupError("Alternative not found")
        current = db.execute(
            "SELECT current_revision_id FROM documents WHERE id = ?",
            (document_id,),
        ).fetchone()
        if current is None:
            raise LookupError("Document not found")
        if current["current_revision_id"] != revision["parent_revision_id"]:
            raise RevisionConflict("The document changed after these alternatives were generated")

        db.execute("UPDATE alternatives SET selected = 0 WHERE revision_id = ?", (revision_id,))
        db.execute("UPDATE alternatives SET selected = 1 WHERE id = ?", (alternative_id,))
        db.execute(
            """
            UPDATE revisions
            SET composite_image_path = ?, selected_alternative_id = ?, accepted = 1
            WHERE id = ?
            """,
            (alternative["output_image_path"], alternative_id, revision_id),
        )
        _set_current_revision(db, document_id, revision_id)
        db.commit()
    return get_document(document_id) or _missing_document(document_id)


def save_layers(document_id: int, base_revision_id: int, layers: list[dict[str, Any]], label: str) -> dict[str, Any]:
    """Records a new layer stack over the current background as an undoable revision."""
    checked = layer_store.validate(layers)
    with storage.connect() as db:
        document = db.execute("SELECT current_revision_id FROM documents WHERE id = ?", (document_id,)).fetchone()
        if document is None:
            raise LookupError("Document not found")
        if document["current_revision_id"] != base_revision_id:
            raise RevisionConflict("The document changed. Try again.")
        base = db.execute("SELECT composite_image_path FROM revisions WHERE id = ?", (base_revision_id,)).fetchone()
        revision_id = _insert_revision(
            db,
            document_id=document_id,
            parent_revision_id=base_revision_id,
            operation_type="layers",
            composite_path=base["composite_image_path"],
            mask_path=None,
            parameters={"label": label},
            accepted=True,
            layers=checked,
        )
        _set_current_revision(db, document_id, revision_id)
        db.commit()
    return get_document(document_id) or _missing_document(document_id)


MIN_CANVAS = 16
MAX_CANVAS = 4096


def crop(document_id: int, base_revision_id: int, x: int, y: int, width: int, height: int) -> dict[str, Any]:
    """Cuts the canvas to a rectangle in background pixels, without AI.

    The rectangle may reach past the picture: that space stays transparent, as
    an empty canvas in Photoshop. Layers keep their place over the content.
    """
    if not (MIN_CANVAS <= width <= MAX_CANVAS and MIN_CANVAS <= height <= MAX_CANVAS):
        raise ValueError(f"The canvas must be {MIN_CANVAS} to {MAX_CANVAS} pixels on each side.")
    with storage.connect() as db:
        document = db.execute("SELECT current_revision_id FROM documents WHERE id = ?", (document_id,)).fetchone()
        if document is None:
            raise LookupError("Document not found")
        if document["current_revision_id"] != base_revision_id:
            raise RevisionConflict("The document changed. Try again.")
        base = db.execute("SELECT composite_image_path, layers FROM revisions WHERE id = ?", (base_revision_id,)).fetchone()
        path = _cropped_image(_absolute_path(base["composite_image_path"]), x, y, width, height)
        try:
            revision_id = _insert_revision(
                db,
                document_id=document_id,
                parent_revision_id=base_revision_id,
                operation_type="crop",
                composite_path=storage.to_relative(path),
                mask_path=None,
                parameters={"x": x, "y": y, "width": width, "height": height},
                accepted=True,
                layers=layer_store.shifted(layer_store.parse(base["layers"]), -x, -y),
            )
            _set_current_revision(db, document_id, revision_id)
            db.commit()
        except BaseException:
            path.unlink(missing_ok=True)
            raise
    return get_document(document_id) or _missing_document(document_id)


def _cropped_image(source: Path, x: int, y: int, width: int, height: int) -> Path:
    with Image.open(source) as image:
        image.load()
        if (x, y, width, height) == (0, 0, *image.size):
            raise ValueError("The canvas already has this size.")
        if x >= image.width or y >= image.height or x + width <= 0 or y + height <= 0:
            raise ValueError("Keep part of the picture on the canvas.")
        if x >= 0 and y >= 0 and x + width <= image.width and y + height <= image.height:
            result = image.crop((x, y, x + width, y + height))
        else:
            result = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            result.paste(image.convert("RGBA"), (-x, -y))
    # PNG, so a cropped JPEG loses nothing more.
    path = settings.revision_dir / f"revision-{uuid4().hex}.png"
    result.save(path, "PNG")
    return path


def delete_document(document_id: int) -> None:
    """Removes a document, all its revisions and associated reference records and files."""
    with storage.connect() as db:
        document = db.execute("SELECT id FROM documents WHERE id = ?", (document_id,)).fetchone()
        if document is None:
            raise LookupError("Document not found")
        revisions = db.execute("SELECT id, composite_image_path, mask_image_path, layers FROM revisions WHERE document_id = ?", (document_id,)).fetchall()
        layer_paths = _revision_layer_paths(revisions)
        rev_ids = [r["id"] for r in revisions]
        alt_paths: list[str] = []
        if rev_ids:
            placeholders = ",".join("?" for _ in rev_ids)
            alternatives = db.execute(f"SELECT output_image_path FROM alternatives WHERE revision_id IN ({placeholders})", rev_ids).fetchall()
            alt_paths = [a["output_image_path"] for a in alternatives]
            db.execute(f"DELETE FROM alternatives WHERE revision_id IN ({placeholders})", rev_ids)
            db.execute("DELETE FROM revisions WHERE document_id = ?", (document_id,))
        refs = []
        if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='document_references'").fetchone():
            refs = db.execute("SELECT image_path, subject_path FROM document_references WHERE document_id = ?", (document_id,)).fetchall()
            db.execute("DELETE FROM document_references WHERE document_id = ?", (document_id,))
        db.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        if layer_paths:
            # Undo and unapplied proposals also keep their layer assets alive.
            remaining_revisions = db.execute("SELECT layers FROM revisions").fetchall()
            layer_paths.difference_update(_revision_layer_paths(remaining_revisions))
        db.commit()

    # Clean up physical files
    files_to_remove: list[str] = []
    for r in revisions:
        if r["composite_image_path"]:
            files_to_remove.append(r["composite_image_path"])
        if r["mask_image_path"]:
            files_to_remove.append(r["mask_image_path"])
    files_to_remove.extend(alt_paths)
    files_to_remove.extend(layer_paths)
    for ref in refs:
        if ref["image_path"]:
            files_to_remove.append(ref["image_path"])
        if ref["subject_path"]:
            files_to_remove.append(ref["subject_path"])

    for relative in files_to_remove:
        try:
            _absolute_path(relative).unlink(missing_ok=True)
        except Exception:
            pass


def _revision_layer_paths(revisions: list[Row]) -> set[str]:
    paths: set[str] = set()
    for revision in revisions:
        for layer in layer_store.parse(revision["layers"]):
            paths.add(layer["image_path"])
            paths.update(layer.get("variant_paths", []))
            if layer.get("mask_path"):
                paths.add(layer["mask_path"])
    return paths


def undo(document_id: int) -> dict[str, Any]:
    with storage.connect() as db:
        row = db.execute(
            """
            SELECT d.current_revision_id, r.parent_revision_id
            FROM documents d
            LEFT JOIN revisions r ON r.id = d.current_revision_id
            WHERE d.id = ?
            """,
            (document_id,),
        ).fetchone()
        if row is None:
            raise LookupError("Document not found")
        if row["parent_revision_id"] is None:
            raise ValueError("Nothing to undo")
        _set_current_revision(db, document_id, int(row["parent_revision_id"]))
        db.commit()
    return get_document(document_id) or _missing_document(document_id)


def redo(document_id: int) -> dict[str, Any]:
    with storage.connect() as db:
        document = db.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
        if document is None:
            raise LookupError("Document not found")
        revision = db.execute(
            """
            SELECT id FROM revisions
            WHERE document_id = ? AND parent_revision_id = ? AND accepted = 1
            ORDER BY id DESC LIMIT 1
            """,
            (document_id, document["current_revision_id"]),
        ).fetchone()
        if revision is None:
            raise ValueError("Nothing to redo")
        _set_current_revision(db, document_id, int(revision["id"]))
        db.commit()
    return get_document(document_id) or _missing_document(document_id)


def get_revision(revision_id: int) -> dict[str, Any] | None:
    with storage.connect() as db:
        row = db.execute("SELECT * FROM revisions WHERE id = ?", (revision_id,)).fetchone()
        return _revision_record(db, row) if row else None


def _document_record(db: Connection, row: Row) -> dict[str, Any]:
    record = dict(row)
    current_revision_id = record["current_revision_id"]
    current = None
    if current_revision_id is not None:
        current = db.execute(
            "SELECT composite_image_path, parent_revision_id, layers FROM revisions WHERE id = ?",
            (current_revision_id,),
        ).fetchone()
    record["current_image_url"] = storage.image_url(current["composite_image_path"]) if current else None
    record["layers"] = layer_store.records(current["layers"]) if current else []
    record["can_undo"] = bool(current and current["parent_revision_id"] is not None)
    record["can_redo"] = bool(
        current_revision_id is not None
        and db.execute(
            """
            SELECT 1 FROM revisions
            WHERE document_id = ? AND parent_revision_id = ? AND accepted = 1
            LIMIT 1
            """,
            (record["id"], current_revision_id),
        ).fetchone()
    )
    return record


def _revision_record(db: Connection, row: Row) -> dict[str, Any]:
    record = dict(row)
    record["accepted"] = bool(record["accepted"])
    record["operation_parameters"] = json.loads(record["operation_parameters"])
    record["composite_image_url"] = storage.image_url(record["composite_image_path"])
    record["layers"] = layer_store.records(record.get("layers"))
    alternatives = db.execute(
        "SELECT * FROM alternatives WHERE revision_id = ? ORDER BY id",
        (record["id"],),
    ).fetchall()
    record["alternatives"] = [_alternative_record(item) for item in alternatives]
    return record


def _alternative_record(row: Row) -> dict[str, Any]:
    record = dict(row)
    record["metadata"] = json.loads(record["metadata"])
    record["selected"] = bool(record["selected"])
    record["output_image_url"] = storage.image_url(record["output_image_path"])
    return record


def _insert_revision(
    db: Connection,
    *,
    document_id: int,
    parent_revision_id: int | None,
    operation_type: str,
    composite_path: str | None,
    mask_path: str | None,
    parameters: dict[str, object],
    accepted: bool,
    layers: list[dict[str, Any]] | None = None,
) -> int:
    if composite_path is None:
        raise ValueError("A revision image is required")
    # Background edits keep the layers that were above them.
    if layers is None:
        layers = _layers_of(db, parent_revision_id)
    cursor = db.execute(
        """
        INSERT INTO revisions
            (document_id, parent_revision_id, operation_type, composite_image_path,
             mask_image_path, operation_parameters, accepted, layers)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            document_id,
            parent_revision_id,
            operation_type,
            composite_path,
            mask_path,
            json.dumps(parameters),
            int(accepted),
            json.dumps(layers),
        ),
    )
    return int(cursor.lastrowid)


def _layers_of(db: Connection, revision_id: int | None) -> list[dict[str, Any]]:
    if revision_id is None:
        return []
    row = db.execute("SELECT layers FROM revisions WHERE id = ?", (revision_id,)).fetchone()
    return layer_store.parse(row["layers"]) if row else []


def _set_current_revision(db: Connection, document_id: int, revision_id: int) -> None:
    revision = db.execute("SELECT composite_image_path FROM revisions WHERE id = ?", (revision_id,)).fetchone()
    if revision is None:
        raise LookupError("Revision not found")
    width, height = _image_size(_absolute_path(revision["composite_image_path"]))
    db.execute(
        "UPDATE documents SET current_revision_id = ?, canvas_width = ?, canvas_height = ?, updated_at = datetime('now') WHERE id = ?",
        (revision_id, width, height, document_id),
    )


def _save_document_image(content: bytes, filename: str | None) -> Path:
    suffix = _safe_suffix(filename)
    path = settings.document_dir / f"document-{uuid4().hex}{suffix}"
    path.write_bytes(content)
    try:
        _image_size(path)
        _apply_orientation(path)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def _apply_orientation(path: Path) -> None:
    """Turns phone photos upright once, as browsers show them, so every tool sees the same pixels."""
    with Image.open(path) as image:
        if image.getexif().get(0x0112, 1) == 1:
            return
        upright = ImageOps.exif_transpose(image)
        exif = upright.getexif()
        exif.pop(0x0112, None)
        options = {"quality": 95, "exif": exif.tobytes()} if image.format == "JPEG" else {}
        upright.save(path, image.format or "PNG", **options)


def _save_revision_image(content: bytes, filename: str | None, prefix: str) -> Path:
    suffix = _safe_suffix(filename)
    path = settings.revision_dir / f"{prefix}-{uuid4().hex}{suffix}"
    path.write_bytes(content)
    _image_size(path)
    return path


def _copy_revision_asset(source: Path, prefix: str) -> Path:
    if not source.is_file():
        raise FileNotFoundError(f"Image asset not found: {source.name}")
    suffix = source.suffix.lower() if source.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"} else ".png"
    destination = settings.revision_dir / f"{prefix}-{uuid4().hex}{suffix}"
    shutil.copy2(source, destination)
    return destination


def _rebased_expansion(generation: dict[str, Any], background: Path, offset: tuple[int, int]) -> Path:
    """Expand's result over the document's own background, so no layer is baked into it."""
    if not generation.get("input_image_path"):
        raise ValueError("This expansion did not keep the picture it continued")
    with (
        Image.open(_absolute_path(generation["output_image_path"])) as result,
        Image.open(_absolute_path(generation["input_image_path"])) as seen,
        Image.open(background) as original,
    ):
        rebased = rebase_expansion(result, seen, original, offset)
    destination = settings.revision_dir / f"alternative-{uuid4().hex}.png"
    rebased.save(destination, "PNG", **FAST_PNG)
    return destination


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        return image.size


def _absolute_path(relative: str) -> Path:
    path = (settings.data_dir / relative).resolve()
    path.relative_to(settings.data_dir.resolve())
    return path


def _safe_suffix(filename: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    return suffix if suffix in {".png", ".jpg", ".jpeg", ".webp"} else ".png"


def _clean_name(name: str, filename: str | None) -> str:
    cleaned = name.strip() or Path(filename or "Untitled").stem or "Untitled"
    return cleaned[:160]


def _missing_document(document_id: int) -> dict[str, Any]:
    raise RuntimeError(f"Document {document_id} disappeared during persistence")


def _missing_revision(revision_id: int) -> dict[str, Any]:
    raise RuntimeError(f"Revision {revision_id} disappeared during persistence")
