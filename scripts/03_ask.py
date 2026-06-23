#!/usr/bin/env python3
"""M3: Ask a question via CLI."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import ask
from src.generate.prompt import ABSTENTION_STRING


def main():
    parser = argparse.ArgumentParser(description="Ask the Pharma Diligence Copilot a question.")
    parser.add_argument("question", help="Question to ask")
    parser.add_argument("--company", help="Filter by ticker (e.g. PFE)")
    parser.add_argument("--year", type=int, help="Filter by fiscal year (e.g. 2024)")
    parser.add_argument("--local", action="store_true", help="Use local Ollama backend (Phase 2)")
    args = parser.parse_args()

    if args.local:
        import os
        os.environ["GENERATOR_BACKEND_OVERRIDE"] = "ollama"

    filters = {}
    if args.company:
        filters["ticker"] = {"$eq": args.company.upper()}
    if args.year:
        filters["fiscal_year"] = {"$eq": args.year}

    result = ask(args.question, filters=filters or None)

    print("\n" + "="*60)
    print(result.answer)
    print("="*60)

    if not result.abstained and result.citations:
        print("\nSources:")
        for c in result.citations:
            print(f"  [{c.n}] {c.company} {c.fiscal_year} 10-K, {c.section}, chunk {c.chunk_id}")

    print(f"\nTiming: {result.timing.total_ms}ms total | Model: {result.model_version}")


if __name__ == "__main__":
    main()
