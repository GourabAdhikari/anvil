"""Persistent ChromaDB-backed key/value memory for Anvil."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_COLLECTION_NAME = "anvil_memory"


def _validate_key(key: str) -> str | None:
    if not isinstance(key, str) or not key.strip():
        return "key must be a non-empty string."
    return None


def _encode(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _decode(value: str) -> Any:
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value


class MemoryStore:
    """Small wrapper around one persistent ChromaDB collection."""

    def __init__(
        self,
        path: str | Path | None = None,
        collection_name: str = _COLLECTION_NAME,
        client: Any = None,
    ) -> None:
        self._path = Path(path or os.getenv("ANVIL_MEMORY_DIR", "~/.anvil/memory")).expanduser()
        self._collection_name = collection_name
        self._client = client
        self._collection: Any = None

    def _get_collection(self) -> Any:
        if self._collection is not None:
            return self._collection
        if self._client is None:
            try:
                import chromadb
            except ImportError as exc:
                raise RuntimeError("ChromaDB is not installed") from exc
            self._path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self._path))
        self._collection = self._client.get_or_create_collection(name=self._collection_name)
        return self._collection

    def remember(self, key: str, value: Any) -> dict[str, Any]:
        """Store or replace a value under ``key``."""
        validation_error = _validate_key(key)
        if validation_error:
            return {"success": False, "error": validation_error}
        try:
            encoded = _encode(value)
            self._get_collection().upsert(ids=[key.strip()], documents=[encoded])
        except Exception as exc:
            return {"success": False, "key": key, "error": str(exc)}
        return {"success": True, "key": key.strip(), "value": value}

    def recall(self, key: str) -> dict[str, Any]:
        """Retrieve the value stored under ``key``."""
        validation_error = _validate_key(key)
        if validation_error:
            return {"success": False, "error": validation_error}
        try:
            result = self._get_collection().get(ids=[key.strip()], include=["documents"])
            documents = result.get("documents") or []
        except Exception as exc:
            return {"success": False, "key": key, "error": str(exc)}
        if not documents:
            return {"success": True, "key": key.strip(), "found": False, "value": None}
        value = _decode(documents[0])
        return {"success": True, "key": key.strip(), "found": True, "value": value}

    def debug_state(self) -> dict[str, Any]:
        """Return stored keys and decoded values for local diagnostics."""
        try:
            result = self._get_collection().get(include=["documents"])
            ids = result.get("ids") or []
            documents = result.get("documents") or []
            values = {key: _decode(document) for key, document in zip(ids, documents)}
            return {"success": True, "collection": self._collection_name, "count": len(values), "values": values}
        except Exception as exc:
            return {"success": False, "collection": self._collection_name, "error": str(exc)}


_default_store: MemoryStore | None = None


def _store() -> MemoryStore:
    global _default_store
    if _default_store is None:
        _default_store = MemoryStore()
    return _default_store


def remember(key: str, value: Any) -> dict[str, Any]:
    """Store a value using the default persistent memory store."""
    return _store().remember(key, value)


def recall(key: str) -> dict[str, Any]:
    """Retrieve a value using the default persistent memory store."""
    return _store().recall(key)


def debug_state() -> dict[str, Any]:
    """Inspect the default memory collection for diagnostics."""
    return _store().debug_state()


__all__ = ["MemoryStore", "remember", "recall", "debug_state"]
