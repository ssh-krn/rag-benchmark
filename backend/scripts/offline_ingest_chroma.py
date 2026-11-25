"""
scripts/offline_ingest_chroma.py

STANDALONE CLI SCRIPT (not an API endpoint, never imported by the FastAPI
app). Run this yourself, offline, to chunk a folder of PDFs/text/markdown
files and load them into a ChromaDB collection that the backend will then
read at query time.

Reuses app.core.embeddings.get_embeddings() rather than instantiating its own
HuggingFaceEmbeddings, so ingestion-time and query-time vectors are produced
by the exact same model + encode_kwargs (mismatched embedding config between
ingestion and querying silently produces garbage similarity scores — there's
no error, just bad retrieval).

Usage:
    python scripts/offline_ingest_chroma.py --input ./docs/ml-foundation \
        --collection ml-foundation

    # Re-ingest from scratch (drops and recreates the collection first):
    python scripts/offline_ingest_chroma.py --input ./docs/ml-foundation \
        --collection ml-foundation --reset

    # Preview chunking without writing anything:
    python scripts/offline_ingest_chroma.py --input ./docs/ml-foundation \
        --collection ml-foundation --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.config import get_settings  # noqa: E402
from app.core.embeddings import get_embeddings  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("offline_ingest_chroma")

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


@dataclass
class RawPage:
    source: str       # original filename, relative to --input
    page: int | None  # 1-indexed page number for PDFs, None for plain text/markdown
    text: str


def _iter_input_files(input_path: Path) -> Iterator[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield input_path
        else:
            logger.warning("Skipping unsupported file type: %s", input_path)
        return

    for path in sorted(input_path.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def _extract_pdf_pages(path: Path) -> list[RawPage]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(RawPage(source=path.name, page=i, text=text))
    return pages


def _extract_plain_text(path: Path) -> list[RawPage]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    return [RawPage(source=path.name, page=None, text=text)] if text else []


def load_pages(input_path: Path) -> Iterable[RawPage]:
    for file_path in _iter_input_files(input_path):
        logger.info("Reading %s", file_path)
        try:
            if file_path.suffix.lower() == ".pdf":
                yield from _extract_pdf_pages(file_path)
            else:
                yield from _extract_plain_text(file_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to read %s: %s", file_path, exc)


def chunk_pages(pages: Iterable[RawPage], chunk_size: int, chunk_overlap: int):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    for page in pages:
        for chunk_index, chunk_text in enumerate(splitter.split_text(page.text)):
            # Deterministic id so re-running ingestion for the same file/page/
            # chunk-index upserts rather than duplicating.
            key = f"{page.source}::{page.page}::{chunk_index}"
            chunk_id = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
            yield chunk_id, chunk_text, {
                "source": page.source,
                "page": page.page,
                "chunk_id": chunk_id,  # read back by app/core/vector_store.py's dense_search
            }


def ingest(input_dir: Path, collection_name: str, chunk_size: int, chunk_overlap: int,
           batch_size: int, reset: bool, dry_run: bool) -> None:
    settings = get_settings()
    pages = list(load_pages(input_dir))
    logger.info("Extracted %d page(s)/document(s) from %s", len(pages), input_dir)

    chunks = list(chunk_pages(pages, chunk_size, chunk_overlap))
    logger.info("Produced %d chunk(s) (chunk_size=%d, overlap=%d)", len(chunks), chunk_size, chunk_overlap)

    if dry_run:
        for chunk_id, text, meta in chunks[:5]:
            logger.info("[dry-run] %s | %s | %r", chunk_id, meta, text[:120])
        logger.info("[dry-run] Would write %d chunks to collection '%s' at %s",
                     len(chunks), collection_name, settings.CHROMA_DIR)
        return

    import chromadb
    from langchain_chroma import Chroma

    client = chromadb.PersistentClient(path=settings.CHROMA_DIR)

    if reset:
        try:
            client.delete_collection(collection_name)
            logger.info("Dropped existing collection '%s'", collection_name)
        except Exception:  # noqa: BLE001
            pass  # collection didn't exist yet — fine

    store = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=get_embeddings(),  # same singleton the backend queries with
    )

    total = len(chunks)
    for start in range(0, total, batch_size):
        batch = chunks[start:start + batch_size]
        ids = [c[0] for c in batch]
        texts = [c[1] for c in batch]
        metadatas = [c[2] for c in batch]
        store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        logger.info("Ingested %d/%d chunks", min(start + batch_size, total), total)

    logger.info("Done. Collection '%s' now has %d chunk(s) at %s",
                collection_name, store._collection.count(), settings.CHROMA_DIR)


def main():
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Offline ChromaDB ingestion (PDF / txt / md)")
    parser.add_argument("--input", required=True, type=Path, help="File or directory to ingest")
    parser.add_argument("--collection", default=settings.DEFAULT_COLLECTION_NAME, help="Target Chroma collection name")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64, help="Chunks per add_texts() call")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate the collection before ingesting")
    parser.add_argument("--dry-run", action="store_true", help="Preview chunking without writing to ChromaDB")
    args = parser.parse_args()

    if not args.input.exists():
        parser.error(f"--input path does not exist: {args.input}")

    ingest(
        input_dir=args.input,
        collection_name=args.collection,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
        reset=args.reset,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
