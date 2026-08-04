"""
Banking RAG Metadata Filter Builder.

Constructs Qdrant metadata payload filter queries based on domain, regulator, section, and document type constraints.
"""

from typing import Dict, Any, Optional, List

from banking_rag.constants import MetadataKeys
from banking_rag.utils.logger import get_logger

logger = get_logger("retrieval.filters")


class MetadataFilterBuilder:
    """Builder for Qdrant payload filters."""

    @staticmethod
    def build_filter(filters: Optional[Dict[str, Any]]) -> Optional[Any]:
        """Constructs a qdrant_client.http.models.Filter object from a simple key-value dictionary.

        Args:
            filters: Dictionary containing key-value constraints (e.g. {'regulator': 'RBI', 'domain': 'AML/KYC'}).

        Returns:
            Qdrant Filter object or None if filters is empty.
        """
        if not filters:
            return None

        try:
            from qdrant_client.http import models

            must_conditions: List[Any] = []

            for key, val in filters.items():
                if val is None or val == "":
                    continue

                if isinstance(val, list):
                    # Match any value in list
                    must_conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchAny(any=val),
                        )
                    )
                else:
                    # Match exact value
                    must_conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=val),
                        )
                    )

            if not must_conditions:
                return None

            logger.info(f"Built Qdrant query filter with {len(must_conditions)} conditions.")
            return models.Filter(must=must_conditions)

        except ImportError:
            logger.warning("qdrant_client not available for Filter building. Returning raw dict.")
            return filters
        except Exception as e:
            logger.error(f"Error building Qdrant payload filter: {str(e)}")
            return None
