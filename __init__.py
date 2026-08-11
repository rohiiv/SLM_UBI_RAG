"""
Union Bank of India Banking RAG Package.
"""

from pathlib import Path as _Path

# Auto-load .env from the package root so config.py's os.getenv() calls pick up
# VECTOR_DB_MODE, MODEL_DEVICE, etc. without the user having to export them manually.
try:
    from dotenv import load_dotenv as _load_dotenv
    _env_path = _Path(__file__).resolve().parent / ".env"
    if _env_path.exists():
        _load_dotenv(_env_path, override=False)
except ImportError:
    pass

__version__ = "1.0.0"
