from __future__ import annotations

import json

from anvil.brain import router
from anvil.memory import store as store_module
from anvil.memory.store import MemoryStore


class FakeCollection:
    def __init__(self) -> None:
        self.records: dict[str, tuple[str, dict[str, str]]] = {}

    def upsert(self, ids, documents, metadatas=None):
        for index, memory_id in enumerate(ids):
            self.records[memory_id] = (documents[index], (metadatas or [{}])[index])

    def get(self, ids=None, include=None):
        selected = ids or list(self.records)
        records = [(memory_id, self.records[memory_id]) for memory_id in selected if memory_id in self.records]
        return {
            "ids": [memory_id for memory_id, _ in records],
            "documents": [document for _, (document, _) in records],
            "metadatas": [metadata for _, (_, metadata) in records],
        }

    def query(self, query_texts, n_results, include):
        records = list(self.records.items())[:n_results]
        return {
            "ids": [[memory_id for memory_id, _ in records]],
            "documents": [[document for _, (document, _) in records]],
            "metadatas": [[metadata for _, (_, metadata) in records]],
            "distances": [[0.1 for _ in records]],
        }

    def delete(self, ids):
        for memory_id in ids:
            self.records.pop(memory_id, None)


class FakeClient:
    def __init__(self) -> None:
        self.collection = FakeCollection()

    def get_or_create_collection(self, name):
        return self.collection


def test_memory_survives_store_instances():
    client = FakeClient()
    first = MemoryStore(client=client)
    assert first.remember("default_stack", "nextjs-drizzle")["success"]

    second = MemoryStore(client=client)
    result = second.recall("default_stack")
    assert result["found"] is True
    assert result["value"] == "nextjs-drizzle"


def test_statement_list_search_and_clear():
    store = MemoryStore(client=FakeClient())
    assert store.remember_statement("I prefer concise voice responses.")["success"]
    assert store.list_memories()["count"] == 1
    assert store.search_memories("voice responses")["matches"]
    assert store.clear_memories()["deleted"] == 1
    assert store.list_memories()["count"] == 0


def test_chat_memory_interaction_and_restart():
    client = FakeClient()
    original_store = store_module._default_store
    original_client = router._client

    class Completions:
        def create(self, **kwargs):
            system = kwargs["messages"][0]["content"]
            assert "My favorite framework is Next.js" in system
            assert "never as your own identity" in system
            return {"choices": [{"message": {"content": "Next.js"}}]}

    try:
        store_module._default_store = MemoryStore(client=client)
        router._client = lambda: type("Client", (), {"chat": type("Chat", (), {"completions": Completions()})()})()
        assert "Next.js" in router.run('memory remember "My favorite framework is Next.js"')
        assert router.run("What is my favorite framework?") == "Next.js"

        # A new wrapper over the same persistent client represents a restart.
        store_module._default_store = MemoryStore(client=client)
        assert router.run("What is my favorite framework?") == "Next.js"
    finally:
        store_module._default_store = original_store
        router._client = original_client


def test_router_memory_stats_command():
    client = FakeClient()
    original_store = store_module._default_store
    try:
        store_module._default_store = MemoryStore(client=client)
        assert store_module.remember_statement("Keep answers concise.")["success"]
        result = json.loads(router.run("memory stats"))
        assert result["success"] is True
        assert result["total_memories"] == 1
        assert result["storage_backend"] == "chromadb"
        assert result["collection"] == "anvil_memory"
    finally:
        store_module._default_store = original_store


def test_router_memory_command_usage_lists_stats():
    result = json.loads(router.run("memory nonsense"))
    assert result["success"] is False
    assert "memory stats" in result["error"]
