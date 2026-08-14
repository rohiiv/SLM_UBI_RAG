"""
Banking RAG Golden Test Set Generator.

Builds a golden evaluation benchmark by:
1. Connecting to the Qdrant vector store (reusing QdrantVectorStoreManager) or fallback local dataset files.
2. Sampling ~200 chunks stratified across banking operational domains.
3. Using the fine-tuned Qwen SLM generator to create realistic compliance questions.
4. Generating 20 unanswerable test cases (out-of-scope regulatory topics).
5. Generating 10 near-miss confusable test cases (overlapping vocabulary from different sources).
6. Writing all ~230 cases to eval/golden_set.jsonl matching project schema.

Usage:
    python3 eval/generate_golden_set.py
    python3 eval/generate_golden_set.py --sample-size 200 --output eval/golden_set.jsonl
"""

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure banking_rag package is importable when executed directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(project_root.parent) not in sys.path:
    sys.path.insert(0, str(project_root.parent))

from banking_rag.constants import BankingDomain, BankingRegulator, MetadataKeys
from banking_rag.llm import QwenBankingSLMGenerator
from banking_rag.utils.logger import get_logger, setup_logger
from banking_rag.vectorstore.qdrant_manager import QdrantVectorStoreManager

logger = get_logger("eval.generate_golden_set")


DOMAIN_MAPPING = {
    "COMPLIANCE": BankingDomain.COMPLIANCE.value,
    "RISK": BankingDomain.RISK.value,
    "INTERNAL_AUDIT": BankingDomain.INTERNAL_AUDIT.value,
    "AML_KYC": BankingDomain.AML_KYC.value,
    "BOARD_SECRETARIAT": BankingDomain.BOARD_SECRETARIAT.value,
    "INTERNAL AUDIT": BankingDomain.INTERNAL_AUDIT.value,
    "BOARD SECRETARIAT": BankingDomain.BOARD_SECRETARIAT.value,
    "AML / KYC": BankingDomain.AML_KYC.value,
}


def normalize_domain(domain_raw: Optional[str]) -> str:
    """Normalizes raw domain metadata into standard BankingDomain enum values."""
    if not domain_raw:
        return BankingDomain.COMPLIANCE.value

    domain_str = str(domain_raw).strip()
    domain_upper = domain_str.upper()

    if domain_upper in DOMAIN_MAPPING:
        return DOMAIN_MAPPING[domain_upper]

    for enum_val in BankingDomain:
        if enum_val.value.upper() == domain_upper:
            return enum_val.value

    return domain_str


def clean_question(text: str) -> str:
    """Cleans raw text output from LLM generator to extract a single clean question string."""
    if not text:
        return ""
    text = text.strip()

    # Remove prefix labels like "Question:", "Q:", "1.", "- "
    text = re.sub(r"^(Question|Q|\d+[\.\)]|\-)\s*:?\s*", "", text, flags=re.IGNORECASE)
    text = text.strip("\"'` ")

    # Take the first non-empty line
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        text = lines[0]

    return text.strip("\"'` ")


def fetch_chunks_from_qdrant(vector_store: QdrantVectorStoreManager) -> List[Dict[str, Any]]:
    """Scrolls and fetches all points from the Qdrant vector store collection."""
    client = vector_store._get_client()
    if client == "MOCK":
        logger.warning("Qdrant client is in MOCK mode. Falling back to local data files.")
        return []

    points = []
    try:
        offset = None
        while True:
            try:
                scroll_res = client.scroll(
                    collection_name=vector_store.collection_name,
                    limit=250,
                    with_payload=True,
                    with_vectors=False,
                    offset=offset,
                )
            except TypeError:
                scroll_res = client.scroll(
                    collection_name=vector_store.collection_name,
                    limit=250,
                    with_payload=True,
                    with_vectors=False,
                    page_offset=offset,
                )

            batch_points, next_offset = scroll_res
            if not batch_points:
                break

            for point in batch_points:
                payload = point.payload or {}
                chunk_id = payload.get(MetadataKeys.CHUNK_ID) or payload.get("chunk_id") or str(point.id)
                doc_name = (
                    payload.get(MetadataKeys.DOC_NAME)
                    or payload.get("doc_name")
                    or payload.get("source_file")
                    or payload.get("source")
                    or "unknown_doc.pdf"
                )
                raw_domain = (
                    payload.get(MetadataKeys.DOMAIN)
                    or payload.get("domain")
                    or payload.get("department")
                    or BankingDomain.COMPLIANCE.value
                )
                regulator = (
                    payload.get(MetadataKeys.REGULATOR)
                    or payload.get("regulator")
                    or payload.get("regulatory_body")
                    or BankingRegulator.OTHER.value
                )
                content = payload.get("content") or payload.get("text") or ""

                if content.strip():
                    points.append({
                        "chunk_id": str(chunk_id),
                        "doc_name": str(doc_name),
                        "domain": normalize_domain(raw_domain),
                        "regulator": str(regulator),
                        "content": content.strip(),
                        "payload": payload,
                    })

            if next_offset is None:
                break
            offset = next_offset

        logger.info(f"Fetched {len(points)} chunks from Qdrant collection '{vector_store.collection_name}'")
    except Exception as e:
        logger.warning(f"Failed to scroll chunks from Qdrant collection: {e}")

    return points


def fetch_chunks_from_fallback_files() -> List[Dict[str, Any]]:
    """Loads chunks from local JSONL data files if Qdrant yields no points."""
    data_dir = project_root / "data"
    candidate_files = [
        data_dir / "sample_dry_run.jsonl",
        data_dir / "dataset_ingest_ready.jsonl",
    ]
    parts_dir = data_dir / "parts"
    if parts_dir.exists():
        candidate_files.extend(sorted(parts_dir.glob("part_*.jsonl")))

    chunks = []
    for filepath in candidate_files:
        if not filepath.exists():
            continue
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line_str = line.strip()
                    if not line_str:
                        continue
                    obj = json.loads(line_str)
                    content = obj.get("text") or obj.get("content") or ""
                    if not content.strip():
                        continue
                    chunk_id = obj.get("chunk_id") or obj.get("document_id") or f"fallback_{len(chunks)}"
                    doc_name = obj.get("source_file") or obj.get("source") or obj.get("doc_name") or "fallback_doc.pdf"
                    raw_domain = obj.get("department") or obj.get("domain") or "Compliance"
                    regulator = obj.get("regulatory_body") or obj.get("regulator") or "Reserve Bank of India (RBI)"

                    chunks.append({
                        "chunk_id": str(chunk_id),
                        "doc_name": str(doc_name),
                        "domain": normalize_domain(raw_domain),
                        "regulator": str(regulator),
                        "content": content.strip(),
                        "payload": obj,
                    })
        except Exception as e:
            logger.warning(f"Error reading fallback file {filepath}: {e}")

    logger.info(f"Loaded {len(chunks)} fallback chunks from local JSONL files.")
    return chunks


def stratify_and_sample_chunks(chunks: List[Dict[str, Any]], sample_size: int) -> List[Dict[str, Any]]:
    """Samples sample_size chunks stratified proportionally across domains."""
    if not chunks:
        logger.error("No chunks available for sampling.")
        return []

    if len(chunks) <= sample_size:
        logger.info(f"Total available chunks ({len(chunks)}) <= sample size ({sample_size}). Returning all chunks.")
        return chunks

    domain_groups = defaultdict(list)
    for c in chunks:
        domain_groups[c["domain"]].append(c)

    sampled_chunks = []
    total_chunks = len(chunks)

    # Compute initial allocation for each domain
    allocations = {}
    total_allocated = 0
    for dom, group in domain_groups.items():
        alloc = max(1, int(round(sample_size * (len(group) / total_chunks))))
        alloc = min(alloc, len(group))
        allocations[dom] = alloc
        total_allocated += alloc

    # Adjust allocation count to match sample_size exactly
    diff = sample_size - total_allocated
    sorted_domains = sorted(domain_groups.keys(), key=lambda d: len(domain_groups[d]), reverse=True)

    while diff != 0:
        changed = False
        for dom in sorted_domains:
            if diff == 0:
                break
            if diff > 0 and allocations[dom] < len(domain_groups[dom]):
                allocations[dom] += 1
                diff -= 1
                changed = True
            elif diff < 0 and allocations[dom] > 1:
                allocations[dom] -= 1
                diff += 1
                changed = True
        if not changed:
            break

    # Perform sampling per domain
    random.seed(42)
    for dom, alloc_count in allocations.items():
        group = domain_groups[dom]
        sampled = random.sample(group, min(alloc_count, len(group)))
        sampled_chunks.extend(sampled)

    domain_counts = {dom: len([c for c in sampled_chunks if c["domain"] == dom]) for dom in domain_groups}
    logger.info(f"Stratified sample of {len(sampled_chunks)} chunks across {len(domain_groups)} domains: {domain_counts}")
    return sampled_chunks


class ProgressTracker:
    """Helper class to log progress every N items generated."""

    def __init__(self, total: int, step: int = 20):
        self.total = total
        self.step = step
        self.count = 0

    def tick(self):
        self.count += 1
        if self.count % self.step == 0 or self.count == self.total:
            logger.info(f"Progress: Generated {self.count}/{self.total} questions ({self.count / self.total * 100:.1f}%)")


def generate_answerable_questions(
    generator: QwenBankingSLMGenerator,
    sampled_chunks: List[Dict[str, Any]],
    tracker: ProgressTracker,
) -> List[Dict[str, Any]]:
    """Generates one realistic compliance question for each sampled chunk."""
    results = []
    system_prompt = (
        "You are a Senior Bank Compliance Officer and Regulatory Auditor. "
        "Formulate clear, precise, realistic compliance questions based on document excerpts."
    )

    for chunk in sampled_chunks:
        user_prompt = (
            "Given this excerpt from a banking/compliance document, write ONE realistic question "
            "a bank compliance officer might ask that this excerpt directly answers. "
            "Return ONLY the question, no preamble.\n\n"
            f"Context excerpt:\n{chunk['content']}"
        )
        try:
            raw_response = generator.generate({"system": system_prompt, "user": user_prompt})
            question = clean_question(raw_response)
        except Exception as e:
            logger.warning(f"LLM generation failed for chunk {chunk['chunk_id']}: {e}")
            question = f"What regulatory rules apply to {chunk['doc_name']}?"

        results.append({
            "question": question,
            "expected_source_doc": chunk["doc_name"],
            "expected_source_chunk_id": chunk["chunk_id"],
            "is_answerable": True,
            "domain": chunk["domain"],
            "regulator": chunk["regulator"],
        })

        tracker.tick()

    return results


def generate_unanswerable_questions(
    generator: QwenBankingSLMGenerator,
    count: int,
    tracker: ProgressTracker,
) -> List[Dict[str, Any]]:
    """Generates out-of-scope / unanswerable compliance questions."""
    unanswerable_topics = [
        "international maritime cargo tax penalties under Panamanian registry",
        "Icelandic geothermal cryptocurrency mining environmental permits",
        "Japanese traditional wooden building municipal property tax exemptions",
        "European Union organic dairy farming subsidy clawback rules",
        "Australian open-cut iron ore mining safety accreditation standards",
        "Brazilian Amazon sustainable timber export quota licensing",
        "Swiss private banking secret numbered accounts legislation of 1934",
        "FAA commercial drone delivery flight corridor airspace authorization",
        "UK social housing tenant dispute ombudsman compensation caps",
        "Canadian Arctic offshore drilling environmental liability guarantees",
        "South African citrus fruit export tariff exemptions under AGOA",
        "Norwegian fjord salmon aquaculture cage density environmental limits",
        "Singapore port marine fuel sulfur emission compliance penalties",
        "Mexican artisanal tequila denomination of origin export certification",
        "German solar energy feed-in tariff historic rate recalculations",
        "Dubai real estate off-plan development escrow account liquidity requirements",
        "South Korean mobile game loot box probability disclosure regulations",
        "Chilean lithium extraction groundwater consumption quota limits",
        "New Zealand biosecurity timber treatment import clearance fees",
        "Kenyan mobile money agent transaction tax withholding obligations",
    ]

    system_prompt = (
        "You are a Banking and Regulatory Specialist. "
        "Create realistic regulatory compliance questions on specific non-banking or foreign topics."
    )

    results = []
    for i in range(count):
        topic = unanswerable_topics[i % len(unanswerable_topics)]
        user_prompt = (
            f"Generate ONE realistic, professional compliance question about {topic}. "
            "Return ONLY the question, no preamble."
        )
        try:
            raw_response = generator.generate({"system": system_prompt, "user": user_prompt})
            question = clean_question(raw_response)
        except Exception as e:
            logger.warning(f"LLM generation failed for unanswerable question on {topic}: {e}")
            question = f"What are the compliance requirements for {topic}?"

        results.append({
            "question": question,
            "expected_source_doc": None,
            "expected_source_chunk_id": None,
            "is_answerable": False,
            "domain": BankingDomain.COMPLIANCE.value,
            "regulator": BankingRegulator.OTHER.value,
        })

        tracker.tick()

    return results


def find_confusable_pairs(chunks: List[Dict[str, Any]], num_pairs: int = 10) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Finds pairs of chunks from different doc_names or domains with keyword vocabulary overlap."""
    keyword_sets = []
    for c in chunks:
        words = set(re.findall(r"\b[a-zA-Z]{4,}\b", c["content"].lower()))
        keyword_sets.append(words)

    pairs = []
    seen = set()
    n = len(chunks)

    for i in range(n):
        if len(pairs) >= num_pairs:
            break
        for j in range(i + 1, n):
            c_a, c_b = chunks[i], chunks[j]
            if c_a["doc_name"] == c_b["doc_name"] and c_a["domain"] == c_b["domain"]:
                continue

            overlap = keyword_sets[i].intersection(keyword_sets[j])
            if len(overlap) >= 4:
                pair_key = (c_a["chunk_id"], c_b["chunk_id"])
                if pair_key not in seen:
                    seen.add(pair_key)
                    pairs.append((c_a, c_b))
                    if len(pairs) >= num_pairs:
                        break

    if len(pairs) < num_pairs and n >= 2:
        for i in range(n - 1):
            if len(pairs) >= num_pairs:
                break
            c_a, c_b = chunks[i], chunks[i + 1]
            if (c_a["chunk_id"], c_b["chunk_id"]) not in seen:
                pairs.append((c_a, c_b))

    return pairs[:num_pairs]


def generate_confusable_questions(
    generator: QwenBankingSLMGenerator,
    chunks: List[Dict[str, Any]],
    count: int,
    tracker: ProgressTracker,
) -> List[Dict[str, Any]]:
    """Generates near-miss confusable questions designed to test retrieval precision."""
    pairs = find_confusable_pairs(chunks, num_pairs=count)
    system_prompt = (
        "You are a Banking Regulatory Retrieval Evaluator. "
        "Formulate precise questions that target specific nuances in Excerpt A to distinguish it from Excerpt B."
    )

    results = []
    for chunk_a, chunk_b in pairs:
        user_prompt = (
            f"Excerpt A (from document '{chunk_a['doc_name']}'):\n{chunk_a['content']}\n\n"
            f"Excerpt B (from document '{chunk_b['doc_name']}'):\n{chunk_b['content']}\n\n"
            "Write ONE specific realistic banking compliance question that is answered ONLY by Excerpt A, "
            "but where Excerpt B could easily be confused as a plausible retrieval candidate due to overlapping terminology. "
            "Focus on details unique to Excerpt A. Return ONLY the question, no preamble."
        )
        try:
            raw_response = generator.generate({"system": system_prompt, "user": user_prompt})
            question = clean_question(raw_response)
        except Exception as e:
            logger.warning(f"LLM generation failed for confusable case: {e}")
            question = f"What specific provisions in {chunk_a['doc_name']} apply regarding compliance requirements?"

        results.append({
            "question": question,
            "expected_source_doc": chunk_a["doc_name"],
            "expected_source_chunk_id": chunk_a["chunk_id"],
            "is_answerable": True,
            "domain": chunk_a["domain"],
            "regulator": chunk_a["regulator"],
            "test_type": "confusable",
        })

        tracker.tick()

    return results


import gc
try:
    import torch
except ImportError:
    torch = None


def clear_memory():
    """Triggers garbage collection and clears PyTorch MPS cache if active."""
    gc.collect()
    if torch is not None and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Generate Golden Evaluation Set for Banking RAG.")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=200,
        help="Number of answerable chunks to sample across domains (default: 200).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(project_root / "eval" / "golden_set.jsonl"),
        help="Output JSONL file path for golden test set (default: eval/golden_set.jsonl).",
    )
    args = parser.parse_args()

    setup_logger()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting Golden Set Generation (sample_size={args.sample_size}, output={output_path})...")

    # 1. Connect to Qdrant vector store
    vector_store = QdrantVectorStoreManager()
    chunks = fetch_chunks_from_qdrant(vector_store)

    # Fallback to local files if Qdrant collection is empty
    if not chunks:
        logger.info("No chunks retrieved from Qdrant vector store. Falling back to local JSONL files...")
        chunks = fetch_chunks_from_fallback_files()

    if not chunks:
        logger.error("No chunks available from vector store or fallback files. Aborting golden set generation.")
        sys.exit(1)

    # 2. Stratified sampling
    sampled_chunks = stratify_and_sample_chunks(chunks, sample_size=args.sample_size)
    num_answerable = len(sampled_chunks)
    num_unanswerable = 20
    num_confusable = 10
    total_target = num_answerable + num_unanswerable + num_confusable

    logger.info(f"Targeting {total_target} total test cases ({num_answerable} answerable + {num_unanswerable} unanswerable + {num_confusable} confusable).")

    # 3. Instantiate LLM generator
    generator = QwenBankingSLMGenerator()
    tracker = ProgressTracker(total=total_target, step=20)

    system_answerable = (
        "You are a Senior Bank Compliance Officer and Regulatory Auditor. "
        "Formulate clear, precise, realistic compliance questions based on document excerpts."
    )
    system_unanswerable = (
        "You are a Banking and Regulatory Specialist. "
        "Create realistic regulatory compliance questions on specific non-banking or foreign topics."
    )
    system_confusable = (
        "You are a Banking Regulatory Retrieval Evaluator. "
        "Formulate precise questions that target specific nuances in Excerpt A to distinguish it from Excerpt B."
    )

    unanswerable_topics = [
        "international maritime cargo tax penalties under Panamanian registry",
        "Icelandic geothermal cryptocurrency mining environmental permits",
        "Japanese traditional wooden building municipal property tax exemptions",
        "European Union organic dairy farming subsidy clawback rules",
        "Australian open-cut iron ore mining safety accreditation standards",
        "Brazilian Amazon sustainable timber export quota licensing",
        "Swiss private banking secret numbered accounts legislation of 1934",
        "FAA commercial drone delivery flight corridor airspace authorization",
        "UK social housing tenant dispute ombudsman compensation caps",
        "Canadian Arctic offshore drilling environmental liability guarantees",
        "South African citrus fruit export tariff exemptions under AGOA",
        "Norwegian fjord salmon aquaculture cage density environmental limits",
        "Singapore port marine fuel sulfur emission compliance penalties",
        "Mexican artisanal tequila denomination of origin export certification",
        "German solar energy feed-in tariff historic rate recalculations",
        "Dubai real estate off-plan development escrow account liquidity requirements",
        "South Korean mobile game loot box probability disclosure regulations",
        "Chilean lithium extraction groundwater consumption quota limits",
        "New Zealand biosecurity timber treatment import clearance fees",
        "Kenyan mobile money agent transaction tax withholding obligations",
    ]

    confusable_pairs = find_confusable_pairs(chunks, num_pairs=num_confusable)
    written_count = 0

    with open(output_path, "w", encoding="utf-8") as f:
        # 4. Generate answerable cases
        logger.info("Generating answerable test cases from sampled chunks...")
        for chunk in sampled_chunks:
            user_prompt = (
                "Given this excerpt from a banking/compliance document, write ONE realistic question "
                "a bank compliance officer might ask that this excerpt directly answers. "
                "Return ONLY the question, no preamble.\n\n"
                f"Context excerpt:\n{chunk['content']}"
            )
            try:
                raw_response = generator.generate({"system": system_answerable, "user": user_prompt})
                question = clean_question(raw_response)
            except Exception as e:
                logger.warning(f"LLM generation failed for chunk {chunk['chunk_id']}: {e}")
                question = f"What regulatory rules apply to {chunk['doc_name']}?"

            case = {
                "question": question,
                "expected_source_doc": chunk["doc_name"],
                "expected_source_chunk_id": chunk["chunk_id"],
                "is_answerable": True,
                "domain": chunk["domain"],
                "regulator": chunk["regulator"],
            }
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
            f.flush()
            written_count += 1
            tracker.tick()
            clear_memory()

        # 5. Generate unanswerable cases
        logger.info("Generating unanswerable test cases...")
        for i in range(num_unanswerable):
            topic = unanswerable_topics[i % len(unanswerable_topics)]
            user_prompt = (
                f"Generate ONE realistic, professional compliance question about {topic}. "
                "Return ONLY the question, no preamble."
            )
            try:
                raw_response = generator.generate({"system": system_unanswerable, "user": user_prompt})
                question = clean_question(raw_response)
            except Exception as e:
                logger.warning(f"LLM generation failed for unanswerable question on {topic}: {e}")
                question = f"What are the compliance requirements for {topic}?"

            case = {
                "question": question,
                "expected_source_doc": None,
                "expected_source_chunk_id": None,
                "is_answerable": False,
                "domain": BankingDomain.COMPLIANCE.value,
                "regulator": BankingRegulator.OTHER.value,
            }
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
            f.flush()
            written_count += 1
            tracker.tick()
            clear_memory()

        # 6. Generate confusable cases
        logger.info("Generating near-miss confusable test cases...")
        for chunk_a, chunk_b in confusable_pairs:
            user_prompt = (
                f"Excerpt A (from document '{chunk_a['doc_name']}'):\n{chunk_a['content']}\n\n"
                f"Excerpt B (from document '{chunk_b['doc_name']}'):\n{chunk_b['content']}\n\n"
                "Write ONE specific realistic banking compliance question that is answered ONLY by Excerpt A, "
                "but where Excerpt B could easily be confused as a plausible retrieval candidate due to overlapping terminology. "
                "Focus on details unique to Excerpt A. Return ONLY the question, no preamble."
            )
            try:
                raw_response = generator.generate({"system": system_confusable, "user": user_prompt})
                question = clean_question(raw_response)
            except Exception as e:
                logger.warning(f"LLM generation failed for confusable case: {e}")
                question = f"What specific provisions in {chunk_a['doc_name']} apply regarding compliance requirements?"

            case = {
                "question": question,
                "expected_source_doc": chunk_a["doc_name"],
                "expected_source_chunk_id": chunk_a["chunk_id"],
                "is_answerable": True,
                "domain": chunk_a["domain"],
                "regulator": chunk_a["regulator"],
                "test_type": "confusable",
            }
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
            f.flush()
            written_count += 1
            tracker.tick()
            clear_memory()

    logger.info(f"Successfully generated {written_count} golden set test cases in {output_path}.")


if __name__ == "__main__":
    main()

