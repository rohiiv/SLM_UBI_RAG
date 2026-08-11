"""
Banking RAG Offline Ingestion Pipeline.

Orchestrates: Document Loading -> Parsing -> Contextual Chunking -> Metadata Extraction -> Embedding -> Qdrant Batch Upsert.
"""

from pathlib import Path
from typing import List, Union, Dict, Any, Optional
import time

from banking_rag.chunking.contextual_chunker import ContextualChunker
from banking_rag.config import IngestionConfig, get_config
from banking_rag.embeddings.embedding_generator import BaseEmbeddingGenerator, BGEEmbeddingGenerator
from banking_rag.exceptions import PipelineError, SecurityError
from banking_rag.loaders.base_loader import BaseDocumentLoader, Document
from banking_rag.loaders.docx_loader import DocxLoader
from banking_rag.loaders.pdf_loader import PDFLoader
from banking_rag.loaders.text_loader import TextLoader
from banking_rag.loaders.jsonl_loader import JSONLLoader
from banking_rag.metadata.metadata_extractor import MetadataExtractor
from banking_rag.parser.parser import DocumentParser
from banking_rag.utils.file_utils import find_documents, validate_file_exists
from banking_rag.utils.logger import get_logger
from banking_rag.vectorstore.qdrant_manager import QdrantVectorStoreManager

logger = get_logger("pipeline.ingest")


def _resolve_allowed_roots(ingestion_config: IngestionConfig) -> List[Path]:
    """Resolves configured allowlist root strings to absolute, real filesystem paths."""
    roots = []
    for raw_root in ingestion_config.allowed_source_roots:
        try:
            roots.append(Path(raw_root).expanduser().resolve())
        except Exception:
            logger.warning(f"Skipping unresolvable ingestion allowlist root: {raw_root!r}")
    return roots


def enforce_ingestion_allowlist(path: Path, ingestion_config: Optional[IngestionConfig] = None) -> None:
    """Rejects ingestion of any path that doesn't resolve under a configured allowed root.

    This is the Layer 1 "source allowlisting" control: for banking/compliance data, ingestion
    must only ever pull from a controlled, versioned set of source locations, never an
    arbitrary path an operator (or a compromised automation script) happens to point the
    ingestion pipeline at. Symlinks are resolved before the check so a symlink placed inside
    an allowed root pointing outside it can't be used to bypass the allowlist.

    Args:
        path: Absolute or relative path to the file being ingested.
        ingestion_config: Optional IngestionConfig; defaults to the process-wide config.

    Raises:
        SecurityError: If the resolved path is not under any allowed root.
    """
    config = ingestion_config or get_config().ingestion
    if not config.enforce_allowlist:
        logger.warning(
            "Ingestion source allowlist enforcement is DISABLED (INGESTION_ENFORCE_ALLOWLIST=false). "
            "Any path on this machine can be ingested into production retrieval."
        )
        return

    resolved_path = path.expanduser().resolve()
    allowed_roots = _resolve_allowed_roots(config)

    if not allowed_roots:
        raise SecurityError(
            "Ingestion allowlist enforcement is enabled but no allowed source roots are "
            "configured (INGESTION_ALLOWED_ROOTS is empty). Refusing to ingest anything."
        )

    for root in allowed_roots:
        try:
            resolved_path.relative_to(root)
            return  # path is under this allowed root - permitted
        except ValueError:
            continue

    raise SecurityError(
        f"Refusing to ingest '{resolved_path}': it does not resolve under any configured "
        f"ingestion allowlist root ({[str(r) for r in allowed_roots]}). Add its containing "
        f"directory to INGESTION_ALLOWED_ROOTS if this source is meant to be trusted."
    )


class OfflineIngestionPipeline:
    """End-to-end Offline Ingestion Pipeline for banking regulatory documents."""

    def __init__(
        self,
        loaders: Optional[List[BaseDocumentLoader]] = None,
        parser: Optional[DocumentParser] = None,
        chunker: Optional[ContextualChunker] = None,
        metadata_extractor: Optional[MetadataExtractor] = None,
        embedding_generator: Optional[BaseEmbeddingGenerator] = None,
        vector_store: Optional[QdrantVectorStoreManager] = None,
    ):
        """Initializes pipeline with injected or default component implementations.

        Args:
            loaders: List of document loader strategies.
            parser: Text normalization parser.
            chunker: Context-aware text chunker.
            metadata_extractor: Metadata tagging extractor.
            embedding_generator: Dense vector embedding generator.
            vector_store: Vector database manager.
        """
        self.loaders = loaders or [PDFLoader(), DocxLoader(), TextLoader(), JSONLLoader()]
        self.parser = parser or DocumentParser()
        self.chunker = chunker or ContextualChunker()
        self.metadata_extractor = metadata_extractor or MetadataExtractor()
        self.embedding_generator = embedding_generator or BGEEmbeddingGenerator()
        self.vector_store = vector_store or QdrantVectorStoreManager()

    def ingest_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Ingests a single document file into Qdrant.

        Args:
            file_path: Path to document file.

        Returns:
            Dictionary summary of ingestion results (chunks ingested, doc_id, execution status).

        Raises:
            PipelineError: If ingestion pipeline step fails.
        """
        path = validate_file_exists(file_path)
        enforce_ingestion_allowlist(path)
        logger.info(f"Starting offline ingestion for file: {path.name}")

        try:
            loader = self._select_loader(path)
            batch_size = get_config().ingestion.batch_size
            doc_stream = loader.load_iter(path)

            total_raw_docs = 0
            total_chunks = 0
            doc_batch: List[Document] = []

            for raw_doc in doc_stream:
                total_raw_docs += 1
                doc_batch.append(raw_doc)

                if len(doc_batch) >= batch_size:
                    chunks_count = self._process_and_upsert_batch(doc_batch)
                    total_chunks += chunks_count
                    print(f"📦 Ingested batch: {total_raw_docs} rows processed ({total_chunks} total chunks in Qdrant)...")
                    doc_batch.clear()

            if doc_batch:
                chunks_count = self._process_and_upsert_batch(doc_batch)
                total_chunks += chunks_count
                print(f"📦 Final batch complete: {total_raw_docs} rows processed ({total_chunks} total chunks in Qdrant)...")
                doc_batch.clear()

            if total_raw_docs == 0:
                logger.warning(f"No content extracted from {path.name}")
                return {"status": "skipped", "chunks": 0}

            summary = {
                "status": "success",
                "file_name": path.name,
                "pages": total_raw_docs,
                "chunks_ingested": total_chunks,
            }
            logger.info(f"Successfully completed ingestion for {path.name}: {summary}")
            return summary

        except Exception as e:
            logger.error(f"Ingestion pipeline failed for file {path.name}: {str(e)}")
            raise PipelineError(f"Ingestion error on {path.name}: {str(e)}")

    def _process_and_upsert_batch(self, raw_docs: List[Document]) -> int:
        """Processes a single batch of documents through parse -> metadata -> chunk -> embed -> upsert."""
        if not raw_docs:
            return 0

        batch_start = time.perf_counter()

        # 1. Parsing & Cleaning
        t0 = time.perf_counter()
        parsed_docs = [self.parser.parse(doc) for doc in raw_docs]
        logger.debug(f"[Timing] Parsing:            {time.perf_counter() - t0:.2f}s ({len(raw_docs)} docs)")

        # 2. Metadata Extraction
        t0 = time.perf_counter()
        for doc in parsed_docs:
            doc.metadata = self.metadata_extractor.extract_metadata(doc)
        logger.debug(f"[Timing] Metadata Extraction: {time.perf_counter() - t0:.2f}s")

        # 3. Contextual Chunking
        t0 = time.perf_counter()
        chunks = self.chunker.process_batch(parsed_docs)
        logger.debug(f"[Timing] Chunking:            {time.perf_counter() - t0:.2f}s -> {len(chunks)} chunks")
        if not chunks:
            return 0

        # 4. Dense Embedding Generation
        t0 = time.perf_counter()
        chunk_texts = [c.content for c in chunks]
        embeddings = self.embedding_generator.generate_embeddings(chunk_texts)
        logger.info(f"[Timing] Embedding:           {time.perf_counter() - t0:.2f}s ({len(chunk_texts)} texts, device={getattr(getattr(self.embedding_generator, 'config', None), 'device', '?')})")

        # 5. Qdrant Storage Batch Upsert
        t0 = time.perf_counter()
        self.vector_store.upsert_chunks(chunks=chunks, embeddings=embeddings)
        logger.info(f"[Timing] Qdrant Upsert:       {time.perf_counter() - t0:.2f}s ({len(chunks)} chunks)")

        logger.debug(f"[Timing] Batch total:         {time.perf_counter() - batch_start:.2f}s")
        return len(chunks)

    def ingest_directory(self, directory: Union[str, Path], recursive: bool = True) -> List[Dict[str, Any]]:
        """Scans and ingests all supported documents in a directory.

        Args:
            directory: Directory path string or Path object.
            recursive: Whether to scan subdirectories.

        Returns:
            List of ingestion summary dictionaries for all processed files.
        """
        # Fail fast on the directory itself before scanning it, rather than failing
        # once per file below.
        enforce_ingestion_allowlist(Path(directory))

        doc_files = find_documents(directory=directory, recursive=recursive)
        logger.info(f"Found {len(doc_files)} target documents for directory ingestion.")

        results = []
        for file_path in doc_files:
            try:
                res = self.ingest_file(file_path)
                results.append(res)
            except Exception as e:
                logger.error(f"Ingestion failed for {file_path}: {str(e)}")
                results.append({"status": "failed", "file_name": file_path.name, "error": str(e)})

        return results

    def _select_loader(self, file_path: Path) -> BaseDocumentLoader:
        """Selects the first loader that supports the file type."""
        for loader in self.loaders:
            if loader.supports(file_path):
                return loader
        raise PipelineError(f"No matching loader registered for file type: {file_path.suffix}")
