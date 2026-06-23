#!/usr/bin/env python3
"""M2: Chunk, embed, and store filings in ChromaDB."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.index import chunk as chunker, embed, store
from src.logging_setup import get_logger
from src.schemas import Filing

log = get_logger("02_index")


def run(verify: bool = False):
    processed_dir = ROOT / "data" / "processed"
    filing_files = list(processed_dir.rglob("*.json"))

    if not filing_files:
        print("No processed filings found. Run 01_ingest.py first.")
        sys.exit(1)

    print(f"Found {len(filing_files)} filing(s) to index.")

    all_chunks = []
    for path in filing_files:
        data = json.loads(path.read_text())
        filing = Filing(**data)
        chunks = chunker.chunk_filing(filing)
        all_chunks.extend(chunks)
        log.info("chunked_filing", ticker=filing.ticker, chunks=len(chunks))

    print(f"Total chunks to embed: {len(all_chunks)}")

    batch_size = 128
    all_embeddings = []
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        texts = [c.text for c in batch]
        embeddings = embed.embed_documents(texts)
        all_embeddings.extend(embeddings)
        print(f"  Embedded {min(i + batch_size, len(all_chunks))}/{len(all_chunks)}", end="\r")
    print()

    store.upsert_chunks(all_chunks, all_embeddings)
    total = store.chunk_count()
    print(f"\nIndex complete. Total chunks in store: {total}")

    if verify:
        print("\nVerification:")
        print(f"  Chunk count: {total}")
        sample = store.query(embed.embed_query("risk factors"), top_k=3)
        print(f"  Sample query returned {len(sample)} result(s)")
        if sample:
            print(f"  Top result: {sample[0]['metadata'].get('company')} — {sample[0]['metadata'].get('section')}")
        print("  Re-run idempotency: upserting same chunks again (should add 0 new)...")
        store.upsert_chunks(all_chunks, all_embeddings)
        total_after = store.chunk_count()
        if total_after != total:
            print(f"  WARNING: chunk count changed {total} -> {total_after}")
            sys.exit(1)
        print(f"  Count unchanged at {total}. M2 verification passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    run(verify=args.verify)
