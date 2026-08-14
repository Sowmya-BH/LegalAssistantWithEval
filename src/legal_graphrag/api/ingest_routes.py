"""
Wraps the UNMODIFIED ingestion/pdf_pipeline.run_pipeline() as an upload
endpoint. No changes to that module.

run_pipeline() is synchronous and can take a while (pdfplumber extraction,
OCR fallback, embedding) — this runs it via BackgroundTasks so the upload
request returns immediately with a job_id, and the frontend polls
GET /api/ingest/{job_id}.

The collections list uses its OWN chromadb client pointed at the same
CHROMA_PERSIST_DIR constant pdf_pipeline.py already exports, instead of
adding a new function to resources.py — keeps every existing pipeline file
byte-for-byte unmodified.
"""

from __future__ import annotations

import re
import tempfile
import uuid
from pathlib import Path

import chromadb
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from ..ingestion.pdf_pipeline import CHROMA_PERSIST_DIR, run_pipeline
from .jobs import get_ingest_job, register_ingest_job, update_ingest_job
from .schemas import CollectionsResponse, IngestJobResponse

router = APIRouter(prefix="/api/ingest", tags=["ingest"])


def _run_ingest_job(job_id: str, pdf_path: str, collection_name: str | None) -> None:
    update_ingest_job(job_id, status="running")
    try:
        result = run_pipeline(pdf_path, collection_name=collection_name)
        update_ingest_job(job_id, status="done", collection_name=result["collection_name"], result=result)
    except Exception as exc:  # noqa: BLE001 - surface any failure to the polling client
        update_ingest_job(job_id, status="error", error=str(exc))
    finally:
        Path(pdf_path).unlink(missing_ok=True)


@router.post("", response_model=IngestJobResponse)
async def start_ingest(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    collection_name: str | None = None,
) -> IngestJobResponse:
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=422, detail="Only .pdf files are supported")

    job_id = str(uuid.uuid4())
    register_ingest_job(job_id, file.filename)

    suffix = re.sub(r"\W+", "_", file.filename)
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{suffix}") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    background_tasks.add_task(_run_ingest_job, job_id, tmp_path, collection_name)

    return IngestJobResponse(job_id=job_id, status="pending", filename=file.filename, collection_name=collection_name)


@router.get("/{job_id}", response_model=IngestJobResponse)
def get_ingest_status(job_id: str) -> IngestJobResponse:
    job = get_ingest_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return IngestJobResponse(job_id=job_id, **job)


@router.get("/meta/collections", response_model=CollectionsResponse)
def list_collections() -> CollectionsResponse:
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return CollectionsResponse(collections=[c.name for c in client.list_collections()])
