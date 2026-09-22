"""FastAPI + SQLite store for viewer notes, pins, traces, and TADs.

The React app talks to these routes through Vite's `/api` proxy. Later
homeworks can write classifier scores into the empty `predictions` table
without changing this schema.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Docker Compose sets this to /data/annotations.db on a bind-mounted volume.
DB_PATH = os.environ.get("SQLITE_PATH", "/data/annotations.db")


def connect() -> sqlite3.Connection:
    """Opens SQLite with Row mapping so SELECT results have column names."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Creates the annotation tables if this is the first run on this volume."""
    db_dir = Path(DB_PATH).parent
    db_dir.mkdir(parents=True, exist_ok=True)
    conn = connect()
    try:
        # One row holds the Notebook tab pad for the whole dataset.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS notebook (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                text TEXT NOT NULL DEFAULT ''
            )
            """
        )
        conn.execute(
            "INSERT OR IGNORE INTO notebook (id, text) VALUES (1, '')"
        )
        # Per-cell free-text notes, keyed the same way as the gallery catalog.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cell_notes (
                replicate INTEGER NOT NULL,
                cell_id TEXT NOT NULL,
                text TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (replicate, cell_id)
            )
            """
        )
        # Optional pinned locus on the 3D cell (Spot_ID plus the pin panel).
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pins (
                replicate INTEGER NOT NULL,
                cell_id TEXT NOT NULL,
                spot_id TEXT,
                trace_id TEXT,
                chrom_start REAL,
                lamina REAL,
                nucleolus REAL,
                PRIMARY KEY (replicate, cell_id)
            )
            """
        )
        # Empty scaffold: later homeworks store a predicted cell type here.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                replicate INTEGER NOT NULL,
                cell_id TEXT NOT NULL,
                predicted_type TEXT,
                confidence REAL,
                PRIMARY KEY (replicate, cell_id)
            )
            """
        )
        # One free-text label per chr19 fiber (Trace_ID) on one catalog cell.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trace_labels (
                replicate INTEGER NOT NULL,
                cell_id TEXT NOT NULL,
                trace_id TEXT NOT NULL,
                text TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (replicate, cell_id, trace_id)
            )
            """
        )
        # Drops the unused dataset-wide TAD slot table from the rejected 1-50 UI.
        conn.execute("DROP TABLE IF EXISTS tad_positions")
        # Per-cell TAD notes: TAD number, genomic start/end, then free text.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tad_notes (
                replicate INTEGER NOT NULL,
                cell_id TEXT NOT NULL,
                chrom_start INTEGER NOT NULL,
                chrom_end INTEGER,
                position INTEGER NOT NULL,
                text TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (replicate, cell_id, chrom_start)
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Builds the SQLite file before the first request.
    init_db()
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextBody(BaseModel):
    text: str = ""


class PinIn(BaseModel):
    spot_id: str | None = None
    trace_id: str | None = None
    chrom_start: float | None = None
    lamina: float | None = None
    nucleolus: float | None = None


class PinBody(BaseModel):
    pin: PinIn | None = None


class TadNoteBody(BaseModel):
    text: str = ""
    position: int = 1
    chrom_end: int | None = None


def pin_from_row(row: sqlite3.Row | None) -> dict | None:
    """Turns a pins table row into the JSON object the frontend stores."""
    if row is None:
        return None
    return {
        "spot_id": row["spot_id"],
        "trace_id": row["trace_id"],
        "chrom_start": row["chrom_start"],
        "lamina": row["lamina"],
        "nucleolus": row["nucleolus"],
    }


@app.get("/api/health")
def health() -> dict:
    """Reports that the container is up and the database file is reachable."""
    init_db()
    return {"ok": True, "sqlite": DB_PATH}


@app.get("/api/notebook")
def get_notebook() -> dict:
    """Returns the single Notebook-tab pad."""
    conn = connect()
    try:
        row = conn.execute("SELECT text FROM notebook WHERE id = 1").fetchone()
        return {"text": row["text"] if row else ""}
    finally:
        conn.close()


@app.put("/api/notebook")
def put_notebook(body: TextBody) -> dict:
    """Replaces the Notebook-tab pad with the submitted text."""
    conn = connect()
    try:
        conn.execute(
            "UPDATE notebook SET text = ? WHERE id = 1",
            (body.text,),
        )
        conn.commit()
        return {"text": body.text}
    finally:
        conn.close()


@app.delete("/api/notebook")
def delete_notebook() -> dict:
    """Clears the Notebook-tab pad. The singleton row stays so GET still works."""
    conn = connect()
    try:
        conn.execute("UPDATE notebook SET text = '' WHERE id = 1")
        conn.commit()
        return {"text": ""}
    finally:
        conn.close()


@app.get("/api/cell-notes")
def list_cell_notes() -> dict:
    """Returns every per-cell note whose text is not empty."""
    conn = connect()
    try:
        # Selects notes with real text, ordered by replicate then cell id.
        rows = conn.execute(
            """
            SELECT replicate, cell_id, text
            FROM cell_notes
            WHERE TRIM(text) != ''
            ORDER BY replicate, CAST(cell_id AS INTEGER), cell_id
            """
        ).fetchall()
        notes = [
            {
                "replicate": row["replicate"],
                "cell_id": row["cell_id"],
                "text": row["text"],
            }
            for row in rows
        ]
        return {"notes": notes}
    finally:
        conn.close()


@app.get("/api/cells/{replicate}/{cell_id}/note")
def get_cell_note(replicate: int, cell_id: str) -> dict:
    """Returns the text note for one catalog cell, or an empty string."""
    conn = connect()
    try:
        row = conn.execute(
            """
            SELECT text FROM cell_notes
            WHERE replicate = ? AND cell_id = ?
            """,
            (replicate, cell_id),
        ).fetchone()
        return {"text": row["text"] if row else ""}
    finally:
        conn.close()


@app.put("/api/cells/{replicate}/{cell_id}/note")
def put_cell_note(replicate: int, cell_id: str, body: TextBody) -> dict:
    """Writes the text note for one catalog cell, or deletes the row if blank."""
    conn = connect()
    try:
        if not body.text.strip():
            conn.execute(
                """
                DELETE FROM cell_notes
                WHERE replicate = ? AND cell_id = ?
                """,
                (replicate, cell_id),
            )
            conn.commit()
            return {"text": ""}
        conn.execute(
            """
            INSERT INTO cell_notes (replicate, cell_id, text)
            VALUES (?, ?, ?)
            ON CONFLICT(replicate, cell_id) DO UPDATE SET text = excluded.text
            """,
            (replicate, cell_id, body.text),
        )
        conn.commit()
        return {"text": body.text}
    finally:
        conn.close()


@app.delete("/api/cells/{replicate}/{cell_id}/note")
def delete_cell_note(replicate: int, cell_id: str) -> dict:
    """Removes the stored cell note row, if any."""
    conn = connect()
    try:
        conn.execute(
            """
            DELETE FROM cell_notes
            WHERE replicate = ? AND cell_id = ?
            """,
            (replicate, cell_id),
        )
        conn.commit()
        return {"text": ""}
    finally:
        conn.close()


@app.get("/api/cells/{replicate}/{cell_id}/pin")
def get_pin(replicate: int, cell_id: str) -> dict:
    """Returns the saved 3D pin for one cell, or pin: null if none."""
    conn = connect()
    try:
        row = conn.execute(
            """
            SELECT spot_id, trace_id, chrom_start, lamina, nucleolus
            FROM pins
            WHERE replicate = ? AND cell_id = ?
            """,
            (replicate, cell_id),
        ).fetchone()
        return {"pin": pin_from_row(row)}
    finally:
        conn.close()


@app.put("/api/cells/{replicate}/{cell_id}/pin")
def put_pin(replicate: int, cell_id: str, body: PinBody) -> dict:
    """Saves or clears the 3D pin for one cell."""
    conn = connect()
    try:
        if body.pin is None:
            conn.execute(
                """
                DELETE FROM pins
                WHERE replicate = ? AND cell_id = ?
                """,
                (replicate, cell_id),
            )
        else:
            pin = body.pin
            conn.execute(
                """
                INSERT INTO pins (
                    replicate, cell_id, spot_id, trace_id,
                    chrom_start, lamina, nucleolus
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(replicate, cell_id) DO UPDATE SET
                    spot_id = excluded.spot_id,
                    trace_id = excluded.trace_id,
                    chrom_start = excluded.chrom_start,
                    lamina = excluded.lamina,
                    nucleolus = excluded.nucleolus
                """,
                (
                    replicate,
                    cell_id,
                    pin.spot_id,
                    pin.trace_id,
                    pin.chrom_start,
                    pin.lamina,
                    pin.nucleolus,
                ),
            )
        conn.commit()
        return {"pin": None if body.pin is None else body.pin.model_dump()}
    finally:
        conn.close()


@app.get("/api/cells/{replicate}/{cell_id}/traces")
def list_cell_trace_labels(replicate: int, cell_id: str) -> dict:
    """Returns every stored fiber label for one catalog cell."""
    conn = connect()
    try:
        # Selects labels for this replicate and Cell_ID, ordered by Trace_ID.
        rows = conn.execute(
            """
            SELECT trace_id, text
            FROM trace_labels
            WHERE replicate = ? AND cell_id = ?
            ORDER BY CAST(trace_id AS INTEGER), trace_id
            """,
            (replicate, cell_id),
        ).fetchall()
        traces = [
            {"trace_id": row["trace_id"], "text": row["text"]}
            for row in rows
        ]
        return {"traces": traces}
    finally:
        conn.close()


@app.get("/api/cells/{replicate}/{cell_id}/traces/{trace_id}")
def get_trace_label(replicate: int, cell_id: str, trace_id: str) -> dict:
    """Returns the text label for one Trace_ID on one cell, or empty text."""
    conn = connect()
    try:
        row = conn.execute(
            """
            SELECT text FROM trace_labels
            WHERE replicate = ? AND cell_id = ? AND trace_id = ?
            """,
            (replicate, cell_id, trace_id),
        ).fetchone()
        return {"text": row["text"] if row else ""}
    finally:
        conn.close()


@app.put("/api/cells/{replicate}/{cell_id}/traces/{trace_id}")
def put_trace_label(
    replicate: int,
    cell_id: str,
    trace_id: str,
    body: TextBody,
) -> dict:
    """Writes the fiber label, or deletes the row if the text is blank."""
    conn = connect()
    try:
        if not body.text.strip():
            conn.execute(
                """
                DELETE FROM trace_labels
                WHERE replicate = ? AND cell_id = ? AND trace_id = ?
                """,
                (replicate, cell_id, trace_id),
            )
            conn.commit()
            return {"text": ""}
        conn.execute(
            """
            INSERT INTO trace_labels (replicate, cell_id, trace_id, text)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(replicate, cell_id, trace_id)
            DO UPDATE SET text = excluded.text
            """,
            (replicate, cell_id, trace_id, body.text),
        )
        conn.commit()
        return {"text": body.text}
    finally:
        conn.close()


@app.delete("/api/cells/{replicate}/{cell_id}/traces/{trace_id}")
def delete_trace_label(replicate: int, cell_id: str, trace_id: str) -> dict:
    """Removes one fiber label row, if any."""
    conn = connect()
    try:
        conn.execute(
            """
            DELETE FROM trace_labels
            WHERE replicate = ? AND cell_id = ? AND trace_id = ?
            """,
            (replicate, cell_id, trace_id),
        )
        conn.commit()
        return {"text": ""}
    finally:
        conn.close()


@app.get("/api/trace-labels")
def list_trace_labels() -> dict:
    """Returns every non-empty fiber label in the dataset for Notebook."""
    conn = connect()
    try:
        # Selects labeled fibers with real text, then cell identity, then Trace_ID.
        rows = conn.execute(
            """
            SELECT replicate, cell_id, trace_id, text
            FROM trace_labels
            WHERE TRIM(text) != ''
            ORDER BY
                replicate,
                CAST(cell_id AS INTEGER),
                cell_id,
                CAST(trace_id AS INTEGER),
                trace_id
            """
        ).fetchall()
        traces = [
            {
                "replicate": row["replicate"],
                "cell_id": row["cell_id"],
                "trace_id": row["trace_id"],
                "text": row["text"],
            }
            for row in rows
        ]
        return {"traces": traces}
    finally:
        conn.close()


def tad_note_from_row(row: sqlite3.Row) -> dict:
    """Turns one tad_notes row into the JSON object the frontend stores."""
    return {
        "replicate": row["replicate"],
        "cell_id": row["cell_id"],
        "chrom_start": row["chrom_start"],
        "chrom_end": row["chrom_end"],
        "position": row["position"],
        "text": row["text"],
    }


@app.get("/api/cells/{replicate}/{cell_id}/tad-notes")
def list_cell_tad_notes(replicate: int, cell_id: str) -> dict:
    """Returns every TAD note stored for one catalog cell."""
    conn = connect()
    try:
        # Selects TAD notes for this cell, ordered by TAD number then start.
        rows = conn.execute(
            """
            SELECT replicate, cell_id, chrom_start, chrom_end, position, text
            FROM tad_notes
            WHERE replicate = ? AND cell_id = ?
            ORDER BY position, chrom_start
            """,
            (replicate, cell_id),
        ).fetchall()
        return {"tads": [tad_note_from_row(row) for row in rows]}
    finally:
        conn.close()


@app.put("/api/cells/{replicate}/{cell_id}/tad-notes/{chrom_start}")
def put_cell_tad_note(
    replicate: int,
    cell_id: str,
    chrom_start: int,
    body: TadNoteBody,
) -> dict:
    """Writes the TAD note, or deletes the row if the text is blank."""
    conn = connect()
    try:
        if not body.text.strip():
            conn.execute(
                """
                DELETE FROM tad_notes
                WHERE replicate = ? AND cell_id = ? AND chrom_start = ?
                """,
                (replicate, cell_id, chrom_start),
            )
            conn.commit()
            return {
                "chrom_start": chrom_start,
                "chrom_end": body.chrom_end,
                "position": body.position,
                "text": "",
            }
        conn.execute(
            """
            INSERT INTO tad_notes (
                replicate, cell_id, chrom_start, chrom_end, position, text
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(replicate, cell_id, chrom_start) DO UPDATE SET
                chrom_end = excluded.chrom_end,
                position = excluded.position,
                text = excluded.text
            """,
            (
                replicate,
                cell_id,
                chrom_start,
                body.chrom_end,
                body.position,
                body.text,
            ),
        )
        conn.commit()
        return {
            "chrom_start": chrom_start,
            "chrom_end": body.chrom_end,
            "position": body.position,
            "text": body.text,
        }
    finally:
        conn.close()


@app.delete("/api/cells/{replicate}/{cell_id}/tad-notes/{chrom_start}")
def delete_cell_tad_note(
    replicate: int,
    cell_id: str,
    chrom_start: int,
) -> dict:
    """Removes one TAD note row, if any."""
    conn = connect()
    try:
        conn.execute(
            """
            DELETE FROM tad_notes
            WHERE replicate = ? AND cell_id = ? AND chrom_start = ?
            """,
            (replicate, cell_id, chrom_start),
        )
        conn.commit()
        return {"text": ""}
    finally:
        conn.close()


@app.get("/api/tad-notes")
def list_tad_notes() -> dict:
    """Returns every non-empty TAD note in the dataset for Notebook."""
    conn = connect()
    try:
        # Selects TAD notes with real text, then cell identity, then TAD number.
        rows = conn.execute(
            """
            SELECT replicate, cell_id, chrom_start, chrom_end, position, text
            FROM tad_notes
            WHERE TRIM(text) != ''
            ORDER BY
                replicate,
                CAST(cell_id AS INTEGER),
                cell_id,
                position,
                chrom_start
            """
        ).fetchall()
        return {"tads": [tad_note_from_row(row) for row in rows]}
    finally:
        conn.close()
