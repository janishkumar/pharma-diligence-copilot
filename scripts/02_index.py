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
        import math
        from src.config import cfg
        print("\nVerification:")
        print(f"  Chunk count: {total}")

        # 1. Drop/collision detection: store count == unique produced ids.
        produced_ids = {c.chunk_id for c in all_chunks}
        if total != len(produced_ids):
            print(f"  FAIL: store has {total} chunks but {len(produced_ids)} unique ids were produced (drops/collisions)")
            sys.exit(1)
        print(f"  Unique produced ids: {len(produced_ids)} == store count")

        # 2. No stale chunks: ids in the store must equal ids produced this run.
        stored_ids = set(store.get_collection().get()["ids"])
        stale = stored_ids - produced_ids
        if stale:
            print(f"  FAIL: {len(stale)} stale chunk(s) in store not produced this run (rebuild from scratch)")
            sys.exit(1)
        print("  No stale chunks in store")

        # 3. Embedding sanity: correct dim, unit-norm, no NaN, on a sample.
        dim = cfg["embeddings"].get("dim", 384)
        sample_ids = list(produced_ids)[:: max(1, len(produced_ids) // 20)][:20]
        got = store.get_collection().get(ids=sample_ids, include=["embeddings"])
        for vec in got["embeddings"]:
            if len(vec) != dim:
                print(f"  FAIL: embedding dim {len(vec)} != {dim}")
                sys.exit(1)
            norm = math.sqrt(sum(x * x for x in vec))
            if any(math.isnan(x) for x in vec) or not (0.9 < norm < 1.1):
                print(f"  FAIL: embedding not unit-norm/has NaN (norm={norm:.3f})")
                sys.exit(1)
        print(f"  Embeddings OK: dim={dim}, unit-norm, no NaN (sampled {len(sample_ids)})")

        # 4. Cross-pass idempotency: re-derive chunks from disk and confirm the
        # chunk_id set is identical (deterministic chunking), then re-upsert and
        # confirm zero count delta.
        rederived = set()
        for path in filing_files:
            f2 = Filing(**json.loads(path.read_text()))
            rederived.update(c.chunk_id for c in chunker.chunk_filing(f2))
        if rederived != produced_ids:
            print(f"  FAIL: re-derived chunk ids differ (non-deterministic chunking): "
                  f"+{len(rederived - produced_ids)} / -{len(produced_ids - rederived)}")
            sys.exit(1)
        store.upsert_chunks(all_chunks, all_embeddings)
        if store.chunk_count() != total:
            print(f"  FAIL: re-upsert changed count {total} -> {store.chunk_count()}")
            sys.exit(1)
        print(f"  Idempotent: re-derived ids identical, count stable at {total}")

        # 5. Relevance smoke test.
        sample = store.query(embed.embed_query("What are the principal risk factors?"), top_k=3)
        print(f"  Sample query returned {len(sample)} result(s); "
              f"top: {sample[0]['metadata'].get('company')} — {sample[0]['metadata'].get('section')}" if sample else "  WARN: empty query result")
        print("  M2 verification passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    run(verify=args.verify)
