#!/usr/bin/env python3
"""
Builds/refreshes the local RAG KB (data/kb.jsonl) from GitHub + local docs.

Usage:
  python scripts/build_kb.py --config configs/rag_sources.yaml --out data/kb.jsonl
  python scripts/build_kb.py --config ... --out ... --force
"""

from __future__ import annotations
import argparse
import logging
import os
import sys
from pathlib import Path

# --- Ensure THIS repo is first on sys.path (avoid clashing 'app' packages) ---
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger("build_kb")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# Import the builder from this project
try:
    from app.core.rag.build import build_kb_from_config, ensure_kb  # type: ignore
except Exception as e:  # pragma: no cover
    logger.error("Failed importing KB builder from app.core.rag.build: %s", e)
    logger.error("Make sure you're running from the project root and PYTHONPATH includes '.'.")
    sys.exit(2)

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to configs/rag_sources.yaml")
    p.add_argument("--out", required=True, help="Output JSONL file, e.g., data/kb.jsonl")
    p.add_argument("--force", action="store_true", help="Delete output file first, then rebuild")
    args = p.parse_args()

    out_path = Path(args.out)
    if args.force and out_path.exists():
        logger.info("Removing existing %s", out_path)
        out_path.unlink()

    # If you want a one-liner that skips if exists, use ensure_kb:
    #   created = ensure_kb(out_jsonl=args.out, config_path=args.config, skip_if_exists=True)
    #   logger.info("KB %s at %s", "ready" if created else "unchanged", args.out)

    # Otherwise, always (re)build:
    n = build_kb_from_config(config_path=args.config, out_jsonl=args.out)
    logger.info("Wrote %d records to %s", n, args.out)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
