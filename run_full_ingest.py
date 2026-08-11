"""
Full dataset ingestion runner for banking_rag.

Ingests all 10 part files from banking_rag/data/parts/ one at a time.
Designed to be resumable: already-completed parts are skipped based on a
progress log file (ingest_progress.log). Safe to re-run after a crash.

Usage:
    python3 run_full_ingest.py
    python3 run_full_ingest.py --start-from 3   # resume from part 3
    python3 run_full_ingest.py --dry-run         # just validate, don't ingest
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# ── Set BEFORE any torch / sentence_transformers import ────────────────────────────
# PYTORCH_ENABLE_MPS_FALLBACK: ops without a Metal kernel fall back to CPU silently.
# PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0: disable MPS memory cap (let macOS manage).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

# ---------------------------------------------------------------------------
# Ensure banking_rag package is importable (and .env is auto-loaded)
# ---------------------------------------------------------------------------
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(project_root.parent) not in sys.path:
    sys.path.insert(0, str(project_root.parent))

from banking_rag.pipeline.ingest import OfflineIngestionPipeline
from banking_rag.config import get_config
from banking_rag.utils.logger import setup_logger, get_logger

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PARTS_DIR = project_root / "data" / "parts"
PROGRESS_FILE = project_root / "data" / "ingest_progress.json"
LOG_FILE = project_root / "logs" / "full_ingest.log"

logger = get_logger("run_full_ingest")


def load_progress() -> dict:
    """Loads the progress tracker from disk."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {}


def save_progress(progress: dict):
    """Persists progress tracker to disk."""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def get_part_files() -> list[Path]:
    """Returns sorted list of part files."""
    parts = sorted(PARTS_DIR.glob("part_*.jsonl"))
    if not parts:
        print(f"\n❌ No part files found in {PARTS_DIR}")
        print(f"   Make sure you've run split_dataset.py first.")
        sys.exit(1)
    return parts


def print_banner(part_files: list[Path]):
    print("\n" + "=" * 65)
    print("  UNION BANK RAG — FULL DATASET INGESTION")
    print("=" * 65)
    print(f"  Parts directory : {PARTS_DIR}")
    print(f"  Total parts     : {len(part_files)}")
    print(f"  Progress log    : {PROGRESS_FILE}")
    print(f"  Full log        : {LOG_FILE}")
    print("=" * 65 + "\n")


def run_ingest(args):
    # Setup logging
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    config = get_config()
    setup_logger(log_level=config.log_level, log_file=LOG_FILE, console_level="WARNING")

    part_files = get_part_files()
    print_banner(part_files)

    # Load existing progress
    progress = load_progress()

    # Build pipeline ONCE (loads embedding model into MPS memory once)
    print("⏳ Loading BGE-M3 embedding model on MPS...")
    print("   [MPS FIRST-RUN NOTE] If this is the first time running on this machine or")
    print("   after a macOS update, Apple's Metal compiler must build GPU shaders for")
    print("   BGE-M3 (~570M params). This can take 5-30 minutes silently. THE PROCESS")
    print("   IS NOT STUCK — it's compiling GPU kernels. Subsequent runs take ~15s.")
    print("   Watch 'Activity Monitor > GPU History' — you should see GPU usage spike.")
    pipeline = OfflineIngestionPipeline()
    pipeline.embedding_generator.preload()
    print("✅ Embedding model loaded on MPS.\n")

    overall_start = time.perf_counter()
    total_chunks_all_parts = 0
    failed_parts = []

    for idx, part_path in enumerate(part_files, start=1):
        part_name = part_path.name

        # Check if user wants to start from a specific part
        if args.start_from and idx < args.start_from:
            print(f"  ⏭  Skipping {part_name} (--start-from {args.start_from})")
            continue

        # Skip already-completed parts
        if part_name in progress and progress[part_name].get("status") == "success":
            chunks = progress[part_name].get("chunks_ingested", "?")
            print(f"  ✅ {part_name} — already ingested ({chunks} chunks). Skipping.")
            total_chunks_all_parts += progress[part_name].get("chunks_ingested", 0)
            continue

        # --- Ingest this part ---
        print(f"\n[{idx:02d}/{len(part_files)}] Ingesting {part_name}  ({part_path.stat().st_size / 1e6:.1f} MB)")

        if args.dry_run:
            print(f"  🔍 DRY RUN — would ingest {part_name}, skipping actual write.")
            continue

        part_start = time.perf_counter()
        try:
            result = pipeline.ingest_file(part_path)
            elapsed = time.perf_counter() - part_start
            chunks = result.get("chunks_ingested", 0)
            total_chunks_all_parts += chunks

            # Save progress
            progress[part_name] = {
                "status": "success",
                "chunks_ingested": chunks,
                "elapsed_seconds": round(elapsed, 1),
                "completed_at": datetime.utcnow().isoformat(),
            }
            save_progress(progress)

            rate = chunks / elapsed if elapsed > 0 else 0
            print(f"  ✅ Done: {chunks:,} chunks | {elapsed:.0f}s | {rate:.0f} chunks/sec")

        except KeyboardInterrupt:
            print(f"\n⚠️  Interrupted during {part_name}. Progress saved. Re-run to resume.")
            save_progress(progress)
            sys.exit(0)

        except Exception as e:
            elapsed = time.perf_counter() - part_start
            logger.error(f"Failed to ingest {part_name}: {e}")
            print(f"  ❌ FAILED: {part_name} — {e}")
            progress[part_name] = {
                "status": "failed",
                "error": str(e),
                "elapsed_seconds": round(elapsed, 1),
                "failed_at": datetime.utcnow().isoformat(),
            }
            save_progress(progress)
            failed_parts.append(part_name)
            # Continue to next part rather than aborting everything
            continue

    # ---------------------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------------------
    total_elapsed = time.perf_counter() - overall_start
    print("\n" + "=" * 65)
    print("  INGESTION COMPLETE")
    print("=" * 65)
    print(f"  Total chunks ingested : {total_chunks_all_parts:,}")
    print(f"  Total wall-clock time : {total_elapsed / 60:.1f} min")
    if failed_parts:
        print(f"  ⚠️  Failed parts       : {', '.join(failed_parts)}")
        print(f"     Re-run this script to retry failed parts only.")
    else:
        print("  All parts completed successfully! 🎉")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Full dataset ingestion runner for Banking RAG")
    parser.add_argument(
        "--start-from", type=int, default=None, metavar="N",
        help="Start from part number N (e.g. --start-from 3 skips parts 1 & 2)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate parts exist and model loads, but skip actual Qdrant upsert"
    )
    args = parser.parse_args()
    run_ingest(args)


if __name__ == "__main__":
    main()
