"""
Union Bank of India Banking RAG System - Main Entry Point.

Provides CLI interface for offline document ingestion and interactive/programmatic RAG queries.
"""

import argparse
import sys
from pathlib import Path

# Add project root and parent directory to sys.path to support execution via both `python main.py` and `python -m banking_rag.main`
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(project_root.parent) not in sys.path:
    sys.path.insert(0, str(project_root.parent))

from banking_rag.cache.retrieval_cache import RetrievalCacheManager
from banking_rag.config import get_config
from banking_rag.constants import BankingDomain, BankingRegulator
from banking_rag.embeddings.embedding_generator import BGEEmbeddingGenerator
from banking_rag.llm.generator import QwenBankingSLMGenerator
from banking_rag.pipeline.ingest import OfflineIngestionPipeline
from banking_rag.pipeline.rag_pipeline import OnlineRAGPipeline
from banking_rag.prompts.prompt_builder import PromptBuilder
from banking_rag.retrieval.hybrid_retriever import HybridRetriever
from banking_rag.retrieval.reranker import CrossEncoderReranker
from banking_rag.retrieval.retriever import DenseRetriever
from banking_rag.utils.logger import setup_logger, get_logger
from banking_rag.vectorstore.qdrant_manager import QdrantVectorStoreManager

logger = get_logger("main")


def build_rag_pipeline() -> OnlineRAGPipeline:
    """Builds the fully-wired OnlineRAGPipeline for this process.

    Every heavy component (embedding model, reranker, Qwen SLM + tokenizer, Qdrant connection)
    is instantiated here exactly once and then injected into the retriever / pipeline objects.
    Callers should build this a single time at startup and reuse the returned pipeline for every
    subsequent query; nothing in `OnlineRAGPipeline.query()` re-creates any of these components.

    Returns:
        A ready-to-use OnlineRAGPipeline instance with all components already loaded.
    """
    print("Loading embedding model...")
    embedding_generator = BGEEmbeddingGenerator()
    embedding_generator.preload()

    print("Loading reranker...")
    reranker = CrossEncoderReranker()
    reranker.preload()

    print("Loading Qwen...")
    llm_generator = QwenBankingSLMGenerator()
    llm_generator.preload()

    vector_store = QdrantVectorStoreManager()
    vector_store.preload()
    print("Connected to Qdrant.")

    dense_retriever = DenseRetriever(embedding_generator=embedding_generator, vector_store=vector_store)
    retriever = HybridRetriever(dense_retriever=dense_retriever)
    prompt_builder = PromptBuilder()
    cache_manager = RetrievalCacheManager()

    rag = OnlineRAGPipeline(
        retriever=retriever,
        reranker=reranker,
        prompt_builder=prompt_builder,
        llm_generator=llm_generator,
        cache_manager=cache_manager,
    )
    print("System Ready.")
    return rag


def run_ingest(target_path: Path) -> None:
    """Runs offline ingestion on a file or directory.

    Args:
        target_path: Path to file or folder.
    """
    pipeline = OfflineIngestionPipeline()
    if target_path.is_file():
        res = pipeline.ingest_file(target_path)
        print(f"\n[INGEST SUCCESS] File: {res.get('file_name')} | Chunks: {res.get('chunks_ingested')}")
    elif target_path.is_dir():
        results = pipeline.ingest_directory(target_path)
        print(f"\n[INGEST COMPLETE] Processed {len(results)} files.")
    else:
        print(f"Error: Path {target_path} does not exist.")


def run_interactive_query(rag: OnlineRAGPipeline) -> None:
    """Launches interactive CLI terminal query interface.

    Args:
        rag: A pre-built OnlineRAGPipeline (all models already loaded). This object is reused
            for every question asked in the loop below — it is never recreated.
    """
    print("\n=======================================================")
    print(" UNION BANK OF INDIA - BANKING RAG ASSISTANT ")
    print(" Serving: Compliance, Risk, Audit, AML/KYC, Board Sec ")
    print("=======================================================\n")
    print("Type 'exit' or 'quit' to close.\n")

    while True:
        try:
            user_input = input("Enter Compliance / Banking Question: ").strip()
            if not user_input or user_input.lower() in ["exit", "quit"]:
                print("Exiting Banking RAG System. Goodbye.")
                break

            response = rag.query(query_text=user_input, top_k=3)

            print("\n-------------------------------------------------------")
            print(f"ANSWER:\n{response.answer}\n")
            if response.citations:
                print("CITATIONS & SOURCES:")
                for cit in response.citations:
                    print(f"  • {cit}")
            print(f"Cached Result: {response.cached}")
            print("-------------------------------------------------------\n")

        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting.")
            break
        except Exception as e:
            print(f"\n[ERROR]: {str(e)}\n")


def main() -> None:
    """Main CLI argument parser entry point."""
    config = get_config()
    log_file = project_root / "logs" / "banking_rag.log"
    setup_logger(log_level=config.log_level, log_file=log_file, console_level="WARNING")
    logger.info(f"Logging to file: {log_file}")

    parser = argparse.ArgumentParser(description="Union Bank of India Enterprise Banking RAG System")
    subparsers = parser.add_subparsers(dest="command", help="System command mode")

    # Ingest subcommand
    ingest_parser = subparsers.add_parser("ingest", help="Ingest banking document(s)")
    ingest_parser.add_argument("--path", "-p", type=str, required=True, help="Path to document file or directory")

    # Query subcommand
    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument("--question", "-q", type=str, help="Question string")

    args = parser.parse_args()

    if args.command == "ingest":
        # Ingestion doesn't need the query-side models (Qwen/reranker), so it keeps its own
        # lightweight pipeline. OfflineIngestionPipeline still benefits from the shared
        # embedding-model cache if it's constructed in the same process as OnlineRAGPipeline.
        run_ingest(Path(args.path))
    elif args.command == "query":
        if args.question:
            rag = build_rag_pipeline()
            res = rag.query(args.question)
            print(f"\nANSWER:\n{res.answer}\n")
            print("CITATIONS:")
            for c in res.citations:
                print(f"  • {c}")
        else:
            rag = build_rag_pipeline()
            run_interactive_query(rag)
    else:
        # Default to interactive mode if no args passed
        rag = build_rag_pipeline()
        run_interactive_query(rag)


if __name__ == "__main__":
    main()
