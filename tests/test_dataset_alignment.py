"""
Unit tests for JSONL Annotated Dataset alignment with RAG Pipeline.
"""

import json
from pathlib import Path
import pytest

from banking_rag.loaders.jsonl_loader import JSONLLoader
from banking_rag.metadata.metadata_extractor import MetadataExtractor
from banking_rag.constants import MetadataKeys


def test_jsonl_annotated_dataset_mapping(tmp_path: Path):
    """Verifies that JSONLLoader correctly extracts fields from dataset_annotated.jsonl."""
    jsonl_file = tmp_path / "sample_dataset.jsonl"
    records = [
        {
            "relevance": "RELEVANT",
            "relevance_confidence": 0.95,
            "department": "Compliance",
            "regulatory_body": "Reserve Bank of India (RBI)",
            "document_type": "MASTER_DIRECTION",
            "document_id": "90725ab1-4454-4d42-92db-d59f95bca3c3",
            "chunk_id": "chunk_abc123",
            "source_file": "RBI_Master_Direction_2026.pdf",
            "page_number": 5,
            "section_heading": "Section 12 - Customer Due Diligence",
            "publication_date": "2026-07-31",
            "text": "All regulated entities must adhere to strict KYC verification norms.",
        },
        {
            "relevance": "NOT_RELEVANT",
            "department": None,
            "regulatory_body": None,
            "document_type": "OTHER",
            "document_id": "doc_junk_999",
            "chunk_id": "chunk_junk_999",
            "source_file": "junk_notice.pdf",
            "page_number": None,
            "section_heading": "Registration details",
            "publication_date": "2026-05-01",
            "text": "This is non-regulatory generic text that should be skipped during ingestion.",
        },
    ]

    with open(jsonl_file, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    loader = JSONLLoader()
    docs = list(loader.load_iter(jsonl_file))

    # Assert that NOT_RELEVANT record was filtered out
    assert len(docs) == 1

    doc = docs[0]
    assert doc.doc_id == "chunk_abc123"
    assert doc.content == "All regulated entities must adhere to strict KYC verification norms."
    assert doc.page_number == 5
    assert doc.metadata[MetadataKeys.DOC_NAME] == "RBI_Master_Direction_2026.pdf"
    assert doc.metadata[MetadataKeys.REGULATOR] == "Reserve Bank of India (RBI)"
    assert doc.metadata[MetadataKeys.DOMAIN] == "Compliance"
    assert doc.metadata[MetadataKeys.DOC_TYPE] == "MASTER_DIRECTION"
    assert doc.metadata[MetadataKeys.SECTION] == "Section 12 - Customer Due Diligence"
    assert doc.metadata[MetadataKeys.DATE] == "2026-07-31"


def test_metadata_extractor_preserves_annotated_fields(tmp_path: Path):
    """Verifies that MetadataExtractor preserves pre-annotated fields from Document."""
    jsonl_file = tmp_path / "single_doc.jsonl"
    record = {
        "relevance": "RELEVANT",
        "department": "Risk",
        "regulatory_body": "Securities and Exchange Board of India (SEBI)",
        "document_type": "CIRCULAR",
        "chunk_id": "sebi_chunk_456",
        "source_file": "SEBI_LODR_Circular.pdf",
        "section_heading": "Clause 49 - Corporate Governance",
        "publication_date": "2026-06-15",
        "text": "Listed entities shall comply with corporate governance guidelines.",
    }

    with open(jsonl_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    loader = JSONLLoader()
    doc = list(loader.load_iter(jsonl_file))[0]

    extractor = MetadataExtractor()
    final_metadata = extractor.extract_metadata(doc)

    assert final_metadata[MetadataKeys.REGULATOR] == "Securities and Exchange Board of India (SEBI)"
    assert final_metadata[MetadataKeys.DOMAIN] == "Risk"
    assert final_metadata[MetadataKeys.DOC_TYPE] == "CIRCULAR"
    assert final_metadata[MetadataKeys.SECTION] == "Clause 49 - Corporate Governance"
    assert final_metadata[MetadataKeys.DATE] == "2026-06-15"
    assert final_metadata[MetadataKeys.DOC_NAME] == "SEBI_LODR_Circular.pdf"


def test_citation_metadata_scrubbing():
    """Verifies that user-facing citations scrub all internal tags and metadata."""
    from banking_rag.utils.text_utils import scrub_metadata_for_exposure, format_citation

    raw_metadata = {
        MetadataKeys.DOC_NAME: "RBI_KYC_Master_Direction.pdf",
        MetadataKeys.REGULATOR: "Reserve Bank of India (RBI)",
        MetadataKeys.SECTION: "Section 12",
        MetadataKeys.PAGE_NUMBER: 4,
        # Internal fields that MUST NOT leak into user-facing citations
        "relevance": "RELEVANT",
        "relevance_confidence": 0.98,
        "risk_indicator": "CONFIDENTIAL",
        "department_confidence": 0.89,
        "file_hash": "a1b2c3d4e5f6",
        "jsonl_line": 142,
        "source_url": "https://internal.bank.net/private/doc.pdf",
        "chunk_id": "chunk_secret_123",
        "internal_class_tag": "STRICT_RESTRICTED",
    }

    scrubbed = scrub_metadata_for_exposure(raw_metadata)
    
    # Assert public fields are preserved
    assert scrubbed[MetadataKeys.DOC_NAME] == "RBI_KYC_Master_Direction.pdf"
    assert scrubbed[MetadataKeys.REGULATOR] == "Reserve Bank of India (RBI)"
    assert scrubbed[MetadataKeys.SECTION] == "Section 12"
    assert scrubbed[MetadataKeys.PAGE_NUMBER] == 4

    # Assert internal leakage fields are stripped out
    assert "relevance" not in scrubbed
    assert "relevance_confidence" not in scrubbed
    assert "risk_indicator" not in scrubbed
    assert "department_confidence" not in scrubbed
    assert "file_hash" not in scrubbed
    assert "jsonl_line" not in scrubbed
    assert "source_url" not in scrubbed
    assert "chunk_id" not in scrubbed
    assert "internal_class_tag" not in scrubbed

    citation = format_citation(raw_metadata)
    assert "RBI_KYC_Master_Direction.pdf" in citation
    assert "Reserve Bank of India (RBI)" in citation
    assert "CONFIDENTIAL" not in citation
    assert "STRICT_RESTRICTED" not in citation
    assert "chunk_secret_123" not in citation

