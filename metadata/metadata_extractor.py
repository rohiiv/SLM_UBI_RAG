"""
Banking RAG Metadata Extractor module.

Extracts domain, regulator, section, clause, version, and date metadata from banking document contents and filenames.
"""

import re
from datetime import datetime
from typing import Dict, Any, List, Optional

from banking_rag.constants import BankingDomain, BankingRegulator, DocumentType, MetadataKeys
from banking_rag.exceptions import MetadataExtractionError
from banking_rag.loaders.base_loader import Document
from banking_rag.utils.logger import get_logger

logger = get_logger("metadata.metadata_extractor")


class MetadataExtractor:
    """Rule-based metadata extractor for Indian Banking & Regulatory documents."""

    # Regex patterns for regulatory metadata
    DATE_PATTERNS = [
        r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})\b",
        r"\b(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{4})\b",
        r"\b(\d{4}[\/\.-]\d{2}[\/\.-]\d{2})\b",
    ]

    SECTION_PATTERNS = [
        r"(?:Section|Sec\.)\s*(\d+[A-Z]?(?:\([\d\w]+\))*)",
        r"(?:Chapter|Chap\.)\s*([IVXLCDM\d]+)",
        r"(?:Clause|Cl\.)\s*(\d+[\.\d]*\w?)",
    ]

    REGULATOR_KEYWORDS = {
        BankingRegulator.RBI: ["reserve bank of india", "rbi", "master direction", "rbi circular"],
        BankingRegulator.SEBI: ["sebi", "securities and exchange board", "lodr"],
        BankingRegulator.FIU_IND: ["fiu-ind", "financial intelligence unit"],
        BankingRegulator.MHA: ["pmla", "prevention of money laundering"],
        BankingRegulator.MCA: ["companies act", "ministry of corporate affairs", "mca"],
    }

    DOMAIN_KEYWORDS = {
        BankingDomain.COMPLIANCE: ["compliance", "regulatory compliance", "circular", "statutory"],
        BankingDomain.RISK: ["credit risk", "market risk", "basel", "operational risk", "var", "icaap"],
        BankingDomain.INTERNAL_AUDIT: ["internal audit", "concurrent audit", "audit finding", "lfarg"],
        BankingDomain.AML_KYC: ["kyc", "aml", "anti-money laundering", "str", "ctr", "cdd", "beneficial owner"],
        BankingDomain.BOARD_SECRETARIAT: ["board minutes", "resolution", "board meeting", "committee meeting"],
    }

    DOC_TYPE_KEYWORDS = {
        DocumentType.MASTER_DIRECTION: ["master direction", "master circular"],
        DocumentType.CIRCULAR: ["circular", "notification"],
        DocumentType.ACT: ["act, 19", "act, 20", "statute"],
        DocumentType.POLICY: ["policy document", "internal policy"],
        DocumentType.AUDIT_REPORT: ["audit report", "observation"],
        DocumentType.BOARD_MINUTES: ["minutes", "agenda"],
    }

    def extract_metadata(self, document: Document) -> Dict[str, Any]:
        """Extracts comprehensive metadata dictionary from a Document instance.

        Args:
            document: Input Document.

        Returns:
            Dictionary containing extracted metadata fields.

        Raises:
            MetadataExtractionError: If metadata extraction fails.
        """
        try:
            content = document.content or ""
            doc_name = document.metadata.get(MetadataKeys.DOC_NAME, "")
            
            combined_text = f"{doc_name}\n{content[:2000]}"  # Inspect header area

            regulator = self._infer_regulator(combined_text)
            domain = self._infer_domain(combined_text)
            doc_type = self._infer_doc_type(combined_text)
            section = self._extract_section(content)
            chapter = self._extract_chapter(content)
            clause = self._extract_clause(content)
            date = self._extract_date(combined_text)
            version = self._extract_version(combined_text)

            extracted = {
                MetadataKeys.DOC_NAME: doc_name,
                MetadataKeys.REGULATOR: regulator.value if regulator else BankingRegulator.OTHER.value,
                MetadataKeys.DOMAIN: domain.value if domain else BankingDomain.COMPLIANCE.value,
                MetadataKeys.DOC_TYPE: doc_type.value if doc_type else DocumentType.CIRCULAR.value,
                MetadataKeys.SECTION: section or "General",
                MetadataKeys.CHAPTER: chapter or "N/A",
                MetadataKeys.CLAUSE: clause or "N/A",
                MetadataKeys.DATE: date or datetime.now().strftime("%Y-%m-%d"),
                MetadataKeys.VERSION: version or "1.0",
                MetadataKeys.CREATED_AT: datetime.utcnow().isoformat(),
            }

            # Merge extracted fallback metadata with existing document metadata,
            # ensuring pre-annotated non-empty fields in document.metadata take precedence.
            merged_metadata = dict(extracted)
            for k, v in document.metadata.items():
                if v is not None and v != "":
                    merged_metadata[k] = v
            return merged_metadata

        except Exception as e:
            logger.error(f"Metadata extraction error for document {document.doc_id}: {str(e)}")
            raise MetadataExtractionError(f"Failed to extract metadata: {str(e)}")

    def _infer_regulator(self, text: str) -> Optional[BankingRegulator]:
        text_lower = text.lower()
        for regulator, keywords in self.REGULATOR_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                return regulator
        return BankingRegulator.INTERNAL

    def _infer_domain(self, text: str) -> Optional[BankingDomain]:
        text_lower = text.lower()
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                return domain
        return BankingDomain.COMPLIANCE

    def _infer_doc_type(self, text: str) -> Optional[DocumentType]:
        text_lower = text.lower()
        for doc_type, keywords in self.DOC_TYPE_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                return doc_type
        return DocumentType.CIRCULAR

    def _extract_section(self, text: str) -> str:
        match = re.search(r"(?:Section|Sec\.)\s*(\d+[A-Z]?(?:\([\d\w]+\))*)", text, re.IGNORECASE)
        return match.group(1) if match else ""

    def _extract_chapter(self, text: str) -> str:
        match = re.search(r"(?:Chapter|Chap\.)\s*([IVXLCDM\d]+)", text, re.IGNORECASE)
        return match.group(1) if match else ""

    def _extract_clause(self, text: str) -> str:
        match = re.search(r"(?:Clause|Cl\.)\s*(\d+[\.\d]*\w?)", text, re.IGNORECASE)
        return match.group(1) if match else ""

    def _extract_date(self, text: str) -> str:
        for pattern in self.DATE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return ""

    def _extract_version(self, text: str) -> str:
        match = re.search(r"\bv(?:ersion)?\s*([\d\.]+)\b", text, re.IGNORECASE)
        return match.group(1) if match else ""
