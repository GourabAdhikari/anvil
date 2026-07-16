"""Persistent ChromaDB-backed memory for Anvil."""

from __future__ import annotations

import hashlib
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
            self._get_collection().upsert(ids=[key.strip()], documents=[encoded], metadatas=[{"type": "key_value"}])
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

    def remember_statement(self, statement: str) -> dict[str, Any]:
        """Persist a natural-language statement for semantic retrieval."""
        if not isinstance(statement, str) or not statement.strip():
            return {"success": False, "error": "statement must be a non-empty string."}
        text = statement.strip()
        memory_id = "statement_" + hashlib.sha256(text.encode("utf-8")).hexdigest()
        try:
            self._get_collection().upsert(
                ids=[memory_id],
                documents=[text],
                metadatas=[{"type": "statement"}],
            )
        except Exception as exc:
            return {"success": False, "error": str(exc)}
        return {"success": True, "id": memory_id, "statement": text}

    def list_memories(self) -> dict[str, Any]:
        """List all persisted key/value and statement memories."""
        try:
            result = self._get_collection().get(include=["documents", "metadatas"])
            ids = result.get("ids") or []
            documents = result.get("documents") or []
            metadatas = result.get("metadatas") or [{} for _ in ids]
            memories = []
            for memory_id, document, metadata in zip(ids, documents, metadatas):
                kind = (metadata or {}).get("type", "key_value")
                memories.append({"id": memory_id, "type": kind, "value": document if kind == "statement" else _decode(document)})
            return {"success": True, "count": len(memories), "memories": memories}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def search_memories(self, query: str, limit: int = 5) -> dict[str, Any]:
        """Semantically search stored memories using ChromaDB embeddings."""
        if not isinstance(query, str) or not query.strip():
            return {"success": False, "error": "query must be a non-empty string."}
        if limit <= 0:
            return {"success": False, "error": "limit must be positive."}
        try:
            result = self._get_collection().query(
                query_texts=[query.strip()],
                n_results=limit,
                include=["documents", "metadatas", "distances"],
            )
            documents = (result.get("documents") or [[]])[0]
            metadatas = (result.get("metadatas") or [[]])[0]
            distances = (result.get("distances") or [[]])[0]
            ids = (result.get("ids") or [[]])[0]
            matches = []
            for memory_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
                kind = (metadata or {}).get("type", "key_value")
                matches.append({"id": memory_id, "type": kind, "value": document if kind == "statement" else _decode(document), "distance": distance})
            return {"success": True, "query": query.strip(), "matches": matches}
        except Exception as exc:
            return {"success": False, "query": query.strip(), "error": str(exc)}

    def clear_memories(self) -> dict[str, Any]:
        """Delete all memories from the collection."""
        try:
            collection = self._get_collection()
            ids = collection.get().get("ids") or []
            if ids:
                collection.delete(ids=ids)
            return {"success": True, "deleted": len(ids)}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def debug_state(self) -> dict[str, Any]:
        """Return stored keys and decoded values for local diagnostics."""
        state = self.list_memories()
        if not state.get("success"):
            return state
        return {"success": True, "collection": self._collection_name, "count": state["count"], "values": state["memories"]}


_default_store: MemoryStore | None = None


def _store() -> MemoryStore:
    global _default_store
    if _default_store is None:
        _default_store = MemoryStore()
    return _default_store


def remember(key: str, value: Any) -> dict[str, Any]:
    return _store().remember(key, value)


def recall(key: str) -> dict[str, Any]:
    return _store().recall(key)


def remember_statement(statement: str) -> dict[str, Any]:
    return _store().remember_statement(statement)


def list_memories() -> dict[str, Any]:
    return _store().list_memories()


def search_memories(query: str, limit: int = 5) -> dict[str, Any]:
    return _store().search_memories(query, limit)


def clear_memories() -> dict[str, Any]:
    return _store().clear_memories()


def debug_state() -> dict[str, Any]:
    return _store().debug_state()


__all__ = [
    "MemoryStore",
    "remember",
    "recall",
    "remember_statement",
    "list_memories",
    "search_memories",
    "clear_memories",
    "debug_state",
]
