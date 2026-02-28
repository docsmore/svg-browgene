"""
ExecutionMemory — In-memory data store for hold_data/merge_data steps.
Allows steps to store extracted data and later merge/combine them.
"""

import copy
import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("browgene.memory")


class ExecutionMemory:
    """
    Per-execution memory store.
    Steps can hold data under a key, and later merge multiple keys together.
    """

    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}

    def hold(self, key: str, value: Any) -> None:
        """Store a value under a key."""
        self._store[key] = copy.deepcopy(value)
        logger.info(f"Memory HOLD: '{key}' = {json.dumps(value, default=str)[:200]}")

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value by key."""
        return copy.deepcopy(self._store.get(key, default))

    def has(self, key: str) -> bool:
        """Check if a key exists in memory."""
        return key in self._store

    def keys(self) -> List[str]:
        """List all keys in memory."""
        return list(self._store.keys())

    def merge(
        self,
        sources: List[str],
        target: str,
        strategy: str = "combine",
    ) -> Any:
        """
        Merge multiple stored values into a new key.

        Strategies:
          - combine: If all sources are dicts, merge them. If lists, concatenate.
          - zip: Zip list sources together into list of tuples.
          - first_non_null: Take the first non-null value.
          - custom: Caller provides a merge function via params.
        """
        values = [self._store.get(src) for src in sources if src in self._store]
        if not values:
            logger.warning(f"Memory MERGE: no values found for sources {sources}")
            return None

        merged: Any = None

        if strategy == "combine":
            if all(isinstance(v, dict) for v in values):
                merged = {}
                for v in values:
                    merged.update(v)
            elif all(isinstance(v, list) for v in values):
                merged = []
                for v in values:
                    merged.extend(v)
            else:
                merged = values
        elif strategy == "zip":
            list_values = [v if isinstance(v, list) else [v] for v in values]
            merged = list(zip(*list_values))
        elif strategy == "first_non_null":
            for v in values:
                if v is not None:
                    merged = v
                    break
        else:
            merged = values

        self._store[target] = copy.deepcopy(merged)
        logger.info(
            f"Memory MERGE: {sources} → '{target}' (strategy={strategy})"
        )
        return merged

    def to_dict(self) -> Dict[str, Any]:
        """Export the full memory store."""
        return copy.deepcopy(self._store)

    def clear(self) -> None:
        """Clear all stored data."""
        self._store.clear()
        logger.info("Memory CLEARED")

    def delete(self, key: str) -> bool:
        """Remove a specific key."""
        if key in self._store:
            del self._store[key]
            logger.info(f"Memory DELETE: '{key}'")
            return True
        return False
