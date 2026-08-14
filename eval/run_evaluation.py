"""
Banking RAG Evaluation Module.

Loads golden set evaluation benchmark, runs OnlineRAGPipeline queries,
computes retrieval, faithfulness, citation, and abstention metrics (overall & by domain),
and outputs results_summary.json and results_detail.csv.
"""

import argparse
import csv
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# Ensure banking_rag package is importable when executed directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(project_root.parent) not in sys.path:
    sys.path.insert(0, str(project_root.parent))

from banking_rag.pipeline.rag_pipeline import OnlineRAGPipeline
from banking_rag.utils.logger import get_logger

logger = get_logger("eval.run_evaluation")


def _parse_csv_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Parses a CSV row dictionary back into a typed result dictionary for aggregate metric computation."""
    def _parse_bool(val: Any) -> bool:
        if isinstance(val, bool):
            return val
        return str(val).strip().lower() in ("true", "1")

    def _parse_optional_bool(val: Any) -> Optional[bool]:
        if val is None or str(val).strip() == "" or str(val).strip().lower() in ("none", "null"):
            return None
        return _parse_bool(val)

    def _parse_optional_int(val: Any) -> Optional[int]:
        if val is None or str(val).strip() == "" or str(val).strip().lower() in ("none", "null"):
            return None
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return None

    def _parse_float(val: Any, default: float = 0.0) -> float:
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    def _parse_int(val: Any, default: int = 0) -> int:
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return default

    retrieved_raw = row.get("retrieved_chunk_ids", "")
    if isinstance(retrieved_raw, list):
        retrieved_chunk_ids = retrieved_raw
    elif retrieved_raw and isinstance(retrieved_raw, str):
        retrieved_chunk_ids = [cid.strip() for cid in retrieved_raw.split("|") if cid.strip()]
    else:
        retrieved_chunk_ids = []

    return {
        "question": row.get("question", "").strip(),
        "domain": row.get("domain", "Unknown"),
        "regulator": row.get("regulator", "Unknown"),
        "is_answerable": _parse_bool(row.get("is_answerable", True)),
        "expected_source_doc": row.get("expected_source_doc"),
        "expected_source_chunk_id": row.get("expected_source_chunk_id"),
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "retrieval_hit": _parse_optional_bool(row.get("retrieval_hit")),
        "retrieval_rank": _parse_optional_int(row.get("retrieval_rank")),
        "reciprocal_rank": _parse_float(row.get("reciprocal_rank"), 0.0),
        "faithfulness_score": _parse_float(row.get("faithfulness_score"), 1.0),
        "citations_dropped_count": _parse_int(row.get("citations_dropped_count"), 0),
        "abstained": _parse_bool(row.get("abstained", False)),
        "abstention_correct": _parse_bool(row.get("abstention_correct", False)),
        "answer": row.get("answer", ""),
    }


def load_golden_set(file_path: Path) -> List[Dict[str, Any]]:
    """Loads golden set evaluation cases from a JSONL file.

    Args:
        file_path: Path to the golden_set.jsonl file.

    Returns:
        List of test case dictionaries.
    """
    if not file_path.exists():
        logger.error(f"Golden set file not found: {file_path}")
        raise FileNotFoundError(f"Golden set file does not exist: {file_path}")

    cases = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str or line_str.startswith("#"):
                continue
            try:
                data = json.loads(line_str)
                cases.append(data)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON on line {line_num} in {file_path}: {e}")

    logger.info(f"Loaded {len(cases)} test cases from {file_path}")
    return cases


def compute_aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes aggregate evaluation metrics across a list of test result dictionaries.

    Args:
        results: List of per-question evaluation result dictionaries.

    Returns:
        Dictionary of aggregate evaluation metrics.
    """
    if not results:
        return {
            "total_questions": 0,
            "answerable_questions": 0,
            "unanswerable_questions": 0,
            "recall_at_5": 0.0,
            "mrr": 0.0,
            "mean_faithfulness_score": 0.0,
            "mean_citations_dropped_count": 0.0,
            "abstention_accuracy": 0.0,
        }

    total_count = len(results)
    answerable_cases = [r for r in results if r["is_answerable"]]
    unanswerable_cases = [r for r in results if not r["is_answerable"]]

    # Recall@5 and MRR calculated over answerable cases where an expected_source_chunk_id was specified
    retrieval_cases = [
        r
        for r in answerable_cases
        if r.get("expected_source_chunk_id") is not None and str(r.get("expected_source_chunk_id")).strip()
    ]

    if retrieval_cases:
        hits = sum(1 for r in retrieval_cases if r["retrieval_hit"] is True)
        recall_at_5 = hits / len(retrieval_cases)
        mrr = sum(r["reciprocal_rank"] for r in retrieval_cases) / len(retrieval_cases)
    else:
        recall_at_5 = 0.0
        mrr = 0.0

    mean_faithfulness = sum(r["faithfulness_score"] for r in results) / total_count
    mean_citations_dropped = sum(r["citations_dropped_count"] for r in results) / total_count
    abstention_correct_count = sum(1 for r in results if r["abstention_correct"] is True)
    abstention_accuracy = abstention_correct_count / total_count

    return {
        "total_questions": total_count,
        "answerable_questions": len(answerable_cases),
        "unanswerable_questions": len(unanswerable_cases),
        "recall_at_5": round(recall_at_5, 4),
        "mrr": round(mrr, 4),
        "mean_faithfulness_score": round(mean_faithfulness, 4),
        "mean_citations_dropped_count": round(mean_citations_dropped, 4),
        "abstention_accuracy": round(abstention_accuracy, 4),
    }


def print_summary_table(overall: Dict[str, Any], by_domain: Dict[str, Dict[str, Any]]) -> None:
    """Prints a formatted summary metrics table to the console."""
    header_fmt = "{:<20} {:>7} {:>6} {:>7} {:>10} {:>8} {:>14} {:>15} {:>15}"
    row_fmt = "{:<20} {:>7} {:>6} {:>7} {:>10.4f} {:>8.4f} {:>14.4f} {:>15.4f} {:>15.4f}"

    separator = "=" * 110

    print("\n" + separator)
    print(" RAG PIPELINE EVALUATION SUMMARY")
    print(separator)
    print(
        header_fmt.format(
            "Domain", "Total", "Ans", "Unans", "Recall@5", "MRR", "Faithfulness", "Dropped Cites", "Abstention Acc"
        )
    )
    print("-" * 110)

    # Print overall row
    print(
        row_fmt.format(
            "Overall",
            overall["total_questions"],
            overall["answerable_questions"],
            overall["unanswerable_questions"],
            overall["recall_at_5"],
            overall["mrr"],
            overall["mean_faithfulness_score"],
            overall["mean_citations_dropped_count"],
            overall["abstention_accuracy"],
        )
    )

    # Print domain breakdown rows
    for domain_name, metrics in sorted(by_domain.items()):
        print(
            row_fmt.format(
                domain_name[:20],
                metrics["total_questions"],
                metrics["answerable_questions"],
                metrics["unanswerable_questions"],
                metrics["recall_at_5"],
                metrics["mrr"],
                metrics["mean_faithfulness_score"],
                metrics["mean_citations_dropped_count"],
                metrics["abstention_accuracy"],
            )
        )

    print(separator + "\n")


def run_evaluation(
    golden_set_path: Path,
    output_dir: Path,
    top_k: int = 5,
) -> None:
    """Executes evaluation pipeline over golden set test cases and writes summary/detail outputs.

    Args:
        golden_set_path: Path to input golden_set.jsonl.
        output_dir: Path to directory where results_summary.json and results_detail.csv will be saved.
        top_k: Top K retrieval budget to pass to pipeline query.
    """
    logger.info(f"Starting RAG evaluation using golden set: {golden_set_path}")
    cases = load_golden_set(golden_set_path)

    if not cases:
        logger.warning("Golden set is empty. No evaluation test cases to execute.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Instantiate OnlineRAGPipeline once
    logger.info("Instantiating OnlineRAGPipeline for evaluation...")
    pipeline = OnlineRAGPipeline()

    fieldnames = [
        "question",
        "domain",
        "regulator",
        "is_answerable",
        "expected_source_doc",
        "expected_source_chunk_id",
        "retrieved_chunk_ids",
        "retrieval_hit",
        "retrieval_rank",
        "reciprocal_rank",
        "faithfulness_score",
        "citations_dropped_count",
        "abstained",
        "abstention_correct",
        "answer",
    ]

    detail_csv_path = output_dir / "results_detail.csv"

    # Resume support: check if results_detail.csv exists and has rows
    completed_questions = set()
    detailed_results: List[Dict[str, Any]] = []

    if detail_csv_path.exists() and detail_csv_path.stat().st_size > 0:
        with open(detail_csv_path, "r", encoding="utf-8") as f_in:
            reader = csv.DictReader(f_in)
            for row in reader:
                q = row.get("question", "").strip()
                if q:
                    completed_questions.add(q)
                    detailed_results.append(_parse_csv_row(row))

    file_mode = "a" if completed_questions else "w"
    skipped_count = sum(1 for c in cases if c.get("question", "").strip() in completed_questions)

    if skipped_count > 0:
        logger.info(
            f"Found existing {detail_csv_path} with {len(completed_questions)} completed case(s). "
            f"Skipping {skipped_count} case(s) already evaluated."
        )

    # 2. Evaluate each test case and write results incrementally
    with open(detail_csv_path, file_mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if file_mode == "w":
            writer.writeheader()
            f.flush()
            os.fsync(f.fileno())

        for idx, case in enumerate(cases, 1):
            question = case.get("question", "").strip()
            if question in completed_questions:
                continue

            expected_doc = case.get("expected_source_doc")
            expected_chunk_id = case.get("expected_source_chunk_id")
            is_answerable = bool(case.get("is_answerable", True))
            domain = case.get("domain", "Unknown")
            regulator = case.get("regulator", "Unknown")

            logger.info(f"Evaluating case {idx}/{len(cases)} [{domain}]: '{question}'")

            response = pipeline.query(query_text=question, top_k=top_k)

            # Extract retrieved chunk details
            retrieved_chunks = response.retrieved_chunks or []
            retrieved_chunk_ids = [
                item.chunk.chunk_id or str(item.chunk.metadata.get("chunk_id", ""))
                for item in retrieved_chunks
            ]

            # Compute retrieval_hit & retrieval_rank
            retrieval_hit = None
            retrieval_rank = None
            reciprocal_rank = 0.0

            if expected_chunk_id is not None and str(expected_chunk_id).strip():
                exp_id_str = str(expected_chunk_id).strip()
                retrieval_hit = False
                for rank_idx, cid in enumerate(retrieved_chunk_ids, 1):
                    if cid == exp_id_str:
                        retrieval_hit = True
                        retrieval_rank = rank_idx
                        reciprocal_rank = 1.0 / rank_idx
                        break

            # Faithfulness score
            faithfulness_meta = response.metadata.get("faithfulness", {})
            faithfulness_score = float(faithfulness_meta.get("faithfulness_score", 1.0))

            # Citations dropped count
            dropped_citations = response.metadata.get("dropped_hallucinated_citations", [])
            citations_dropped_count = len(dropped_citations)

            # Abstention correctness
            abstained = bool(response.metadata.get("abstained", False))
            if not is_answerable:
                abstention_correct = (abstained is True)
            else:
                abstention_correct = (abstained is False)

            row = {
                "question": question,
                "domain": domain,
                "regulator": regulator,
                "is_answerable": is_answerable,
                "expected_source_doc": expected_doc,
                "expected_source_chunk_id": expected_chunk_id,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "retrieval_hit": retrieval_hit,
                "retrieval_rank": retrieval_rank,
                "reciprocal_rank": reciprocal_rank,
                "faithfulness_score": faithfulness_score,
                "citations_dropped_count": citations_dropped_count,
                "abstained": abstained,
                "abstention_correct": abstention_correct,
                "answer": response.answer,
            }
            detailed_results.append(row)

            row_copy = dict(row)
            if isinstance(row_copy.get("retrieved_chunk_ids"), list):
                row_copy["retrieved_chunk_ids"] = "|".join(row_copy["retrieved_chunk_ids"])
            writer.writerow(row_copy)
            f.flush()
            os.fsync(f.fileno())

            # Memory management: per-iteration cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Deeper periodic cleanup & logging every 20 iterations
            if idx % 20 == 0 and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                alloc_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                res_gb = torch.cuda.memory_reserved() / (1024 ** 3)
                logger.info(
                    f"GPU Memory at iteration {idx}/{len(cases)}: "
                    f"Allocated = {alloc_gb:.2f} GB, Reserved = {res_gb:.2f} GB"
                )

    logger.info(f"Wrote evaluation detail CSV to {detail_csv_path}")

    # 3. Aggregate metrics overall and by domain
    overall_metrics = compute_aggregate_metrics(detailed_results)

    by_domain_results: Dict[str, List[Dict[str, Any]]] = {}
    for r in detailed_results:
        dom = r["domain"] or "Unknown"
        by_domain_results.setdefault(dom, []).append(r)

    by_domain_metrics: Dict[str, Dict[str, Any]] = {
        dom: compute_aggregate_metrics(group) for dom, group in by_domain_results.items()
    }

    summary_payload = {
        "overall": overall_metrics,
        "by_domain": by_domain_metrics,
    }

    # 4. Write summary JSON
    summary_json_path = output_dir / "results_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    logger.info(f"Wrote evaluation summary to {summary_json_path}")

    # 5. Print console summary table
    print_summary_table(overall_metrics, by_domain_metrics)


def main() -> None:
    """CLI entrypoint for running evaluation script."""
    parser = argparse.ArgumentParser(description="Run evaluation on Union Bank RAG Pipeline using Golden Set.")
    parser.add_argument(
        "--golden-set",
        type=str,
        default="eval/golden_set.jsonl",
        help="Path to golden_set.jsonl file (default: eval/golden_set.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval",
        help="Directory to save evaluation results (default: eval)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top K retrieval budget (default: 5)",
    )

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    base_dir = script_dir.parent  # banking_rag root

    golden_set_path = Path(args.golden_set)
    if not golden_set_path.is_absolute():
        golden_set_path = base_dir / golden_set_path

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir

    try:
        run_evaluation(golden_set_path=golden_set_path, output_dir=output_dir, top_k=args.top_k)
    except Exception as e:
        logger.error(f"Evaluation execution failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
