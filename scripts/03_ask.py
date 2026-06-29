#!/usr/bin/env python3
"""M3: Ask a question via CLI."""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# --local must switch the generator backend BEFORE importing the pipeline,
# because generator.BACKEND is resolved at import time.
if "--local" in sys.argv:
    os.environ["GENERATOR_BACKEND_OVERRIDE"] = "ollama"

from src.pipeline import ask  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Ask the Pharma Diligence Copilot a question.")
    parser.add_argument("question", help="Question to ask")
    parser.add_argument("--company", help="Filter by ticker (e.g. PFE)")
    parser.add_argument("--year", type=int, help="Filter by fiscal year (e.g. 2024)")
    parser.add_argument("--local", action="store_true", help="Use local Ollama backend (Phase 2)")
    args = parser.parse_args()

    filters = {}
    if args.company:
        filters["ticker"] = {"$eq": args.company.upper()}
    if args.year:
        filters["fiscal_year"] = {"$eq": args.year}

    result = ask(args.question, filters=filters or None)

    print("\n" + "=" * 60)
    print(result.answer)
    print("=" * 60)

    if result.abstained:
        print("\n[abstained — no grounded answer in the corpus]")
    elif result.citations:
        print("\nSources:")
        for c in result.citations:
            print(f"  [{c.n}] {c.company} {c.fiscal_year} 10-K, {c.section}, chunk {c.chunk_id}")

    if result.truncated:
        print("\n[WARNING: answer was truncated at the token limit — it may be incomplete]")

    print(f"\nTiming: {result.timing.total_ms}ms total | Model: {result.model_version}")


if __name__ == "__main__":
    main()
