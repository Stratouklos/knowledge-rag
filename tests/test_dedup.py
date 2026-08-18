"""Regression tests for chunk dedup integrity (#90, #91) and fuzzy near-duplicate dedup.

#90: Moving an indexed file must not lose chunks on reindex.
#91: Removing a document must not delete chunks shared with another document.

These tests use a real ChromaDB (tmp_path) with mocked embeddings to exercise
the actual indexing pipeline without requiring model downloads.
"""

from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from mcp_server.ingestion import Chunk, Document
from mcp_server.server import KnowledgeOrchestrator, _fuzzy_chunk_hash


class _FakeEmbeddings:
    """Minimal embedding function that satisfies ChromaDB's full interface."""

    _dim = 384
    is_legacy = False

    def __call__(self, input: List[str]) -> List[List[float]]:
        return [[0.1] * 384 for _ in input]

    @staticmethod
    def name() -> str:
        return "fake-test-embeddings"

    @staticmethod
    def build_from_config(config):  # noqa: ARG004
        return _FakeEmbeddings()

    @staticmethod
    def get_config():
        return {}

    @staticmethod
    def validate_config_update(old_config, new_config):  # noqa: ARG004
        pass


@pytest.fixture
def rag_env(tmp_path, monkeypatch):
    """Isolated RAG environment: real ChromaDB, mocked embeddings, tmp dirs."""
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    chroma_dir = data_dir / "chroma_db"
    chroma_dir.mkdir()
    models_dir = tmp_path / "models_cache"
    models_dir.mkdir()

    import mcp_server.config as cfg

    monkeypatch.setattr(cfg.config, "documents_dir", docs_dir)
    monkeypatch.setattr(cfg.config, "data_dir", data_dir)
    monkeypatch.setattr(cfg.config, "chroma_dir", chroma_dir)
    monkeypatch.setattr(cfg.config, "models_cache_dir", models_dir)
    monkeypatch.setattr(cfg.config, "transport", "stdio")

    with patch("mcp_server.server.FastEmbedEmbeddings", _FakeEmbeddings):
        from mcp_server.server import KnowledgeOrchestrator

        orch = KnowledgeOrchestrator()
        yield orch, docs_dir


CONTENT_A = """\
# Document Alpha

## Overview

This document covers advanced SQL injection bypass techniques including
UNION-based, blind boolean, and time-based attacks against web applications.

## Methodology

Use parameterized queries and input validation as primary defenses.
"""

CONTENT_B = """\
# Document Beta

## Overview

Cross-site scripting attack vectors for reflected, stored, and DOM-based
XSS in modern single-page applications with CSP bypass techniques.

## Methodology

Use parameterized queries and input validation as primary defenses.
"""


class TestFileMovePreservesChunks:
    """#90: Moving an indexed file must re-index at the new path."""

    def test_move_file_chunks_preserved(self, rag_env):
        orch, docs_dir = rag_env

        (docs_dir / "alpha.md").write_text(CONTENT_A, encoding="utf-8")
        stats1 = orch.index_all(force=True)
        assert stats1["chunks_added"] > 0
        chunks_before = orch.collection.count()

        sub = docs_dir / "sub"
        sub.mkdir()
        (docs_dir / "alpha.md").rename(sub / "alpha.md")

        stats2 = orch.index_all(force=False)
        chunks_after = orch.collection.count()

        assert chunks_after >= chunks_before, (
            f"Chunks dropped from {chunks_before} to {chunks_after} after move. Orphan cleanup raced dedup (issue #90)."
        )
        assert stats2["deleted"] >= 1

    def test_move_file_searchable_after_reindex(self, rag_env):
        orch, docs_dir = rag_env

        (docs_dir / "alpha.md").write_text(CONTENT_A, encoding="utf-8")
        orch.index_all(force=True)
        chunks_before = len(orch.collection.get(where={"filename": "alpha.md"}, include=[])["ids"])

        sub = docs_dir / "sub"
        sub.mkdir()
        (docs_dir / "alpha.md").rename(sub / "alpha.md")
        orch.index_all(force=False)

        chunks_after = len(orch.collection.get(where={"filename": "alpha.md"}, include=[])["ids"])
        assert chunks_after == chunks_before, (
            f"Chunks went from {chunks_before} to {chunks_after} after move+reindex. "
            "File content should still be fully searchable."
        )


class TestCrossDocDedup:
    """#91: Removing one doc must not delete chunks from another doc."""

    def test_shared_content_survives_removal(self, rag_env):
        orch, docs_dir = rag_env

        (docs_dir / "alpha.md").write_text(CONTENT_A, encoding="utf-8")
        (docs_dir / "beta.md").write_text(CONTENT_B, encoding="utf-8")
        orch.index_all(force=True)

        beta_chunks_before = orch.collection.get(where={"filename": "beta.md"}, include=[])
        assert len(beta_chunks_before["ids"]) > 0

        (docs_dir / "alpha.md").unlink()
        orch.index_all(force=False)

        beta_chunks_after = orch.collection.get(where={"filename": "beta.md"}, include=[])
        assert len(beta_chunks_after["ids"]) == len(beta_chunks_before["ids"]), (
            f"Beta chunks dropped from {len(beta_chunks_before['ids'])} to "
            f"{len(beta_chunks_after['ids'])} after removing alpha. "
            "Cross-document dedup coupling (issue #91)."
        )

    def test_identical_files_deduplicated(self, rag_env):
        """Global fuzzy dedup collapses byte-identical files to one chunk set."""
        orch, docs_dir = rag_env

        (docs_dir / "copy1.md").write_text(CONTENT_A, encoding="utf-8")
        (docs_dir / "copy2.md").write_text(CONTENT_A, encoding="utf-8")
        orch.index_all(force=True)

        c1 = orch.collection.get(where={"filename": "copy1.md"}, include=[])
        c2 = orch.collection.get(where={"filename": "copy2.md"}, include=[])

        # Identical content is stored once; the second file's chunks are deduped.
        assert len(c1["ids"]) + len(c2["ids"]) > 0, "identical content should be indexed at least once"
        assert len(c1["ids"]) == 0 or len(c2["ids"]) == 0, (
            "byte-identical files should not both hold their own chunk set under global dedup"
        )

    def test_remove_one_copy_other_intact(self, rag_env):
        orch, docs_dir = rag_env

        (docs_dir / "copy1.md").write_text(CONTENT_A, encoding="utf-8")
        (docs_dir / "copy2.md").write_text(CONTENT_A, encoding="utf-8")
        orch.index_all(force=True)

        count_before = len(orch.collection.get(where={"filename": "copy2.md"}, include=[])["ids"])

        (docs_dir / "copy1.md").unlink()
        orch.index_all(force=False)

        count_after = len(orch.collection.get(where={"filename": "copy2.md"}, include=[])["ids"])
        assert count_after == count_before, (
            f"copy2 chunks dropped from {count_before} to {count_after} "
            "after removing copy1 — cross-doc dedup coupling."
        )


# ── Fuzzy hash function ──


def test_fuzzy_hash_collapses_whitespace_and_case():
    """Whitespace variation and case must produce the same fuzzy hash."""
    a = "Hello World"
    b = "hello   world"
    c = "HELLO\nworld"
    assert _fuzzy_chunk_hash(a) == _fuzzy_chunk_hash(b) == _fuzzy_chunk_hash(c)


def test_fuzzy_hash_differs_for_distinct_content():
    """Genuinely different content must produce different fuzzy hashes."""
    a = "OpenEBS scraper configuration uses Helm values."
    b = "Mayastor io-engine restart runbook for production."
    assert _fuzzy_chunk_hash(a) != _fuzzy_chunk_hash(b)


def test_fuzzy_hash_collapses_repeated_paragraph_with_trailing_diff():
    """Two chunks with the same opening paragraph but different tails collapse.

    Mirrors the AI-generated architecture-doc pattern where a stock bullet
    list opens both sections; only what comes after the 500-char window
    diverges.
    """
    head = (
        "OpenEBS scraper configuration uses Helm values to define the "
        "scraper deployment. The scraper polls /metrics every 30 seconds "
        "and ships the samples to Prometheus. Replicas default to 3 in "
        "production with the openebs-mayastor storage class. Disk-pressure "
        "alerts fire when usage crosses 85 percent. Pods schedule on "
        "dedicated infra nodes with explicit toleration for the "
        "infra-only taint. The fanout layer batches writes per region and "
        "flushes them in 5-second windows to bound failover blast radius. "
        "Cross-AZ replication is opt-in and gated behind a shard-level flag."
    )
    a = head + " First trailing detail about the alpha shard."
    b = head + " Different trailing detail about the beta shard."
    assert len(head) >= 500, "head must fill the fuzzy window"
    assert _fuzzy_chunk_hash(a) == _fuzzy_chunk_hash(b)


# ── _index_document dedup ──


def _make_orchestrator() -> KnowledgeOrchestrator:
    """Construct an orchestrator with stubbed-out ChromaDB / BM25 / embedder."""
    orch = KnowledgeOrchestrator.__new__(KnowledgeOrchestrator)
    orch.collection = MagicMock()
    orch.bm25_index = MagicMock()
    orch._chunk_hashes = {}
    orch._chunk_fuzzy_hashes = {}
    return orch


def _make_doc(chunks_text):
    chunks = [
        Chunk(content=t, index=i, start_char=0, end_char=len(t), metadata={})
        for i, t in enumerate(chunks_text)
    ]
    return Document(
        id="doc1",
        content="\n\n".join(chunks_text),
        source=Path(__file__),
        format=".md",
        category="general",
        chunks=chunks,
    )


def test_exact_duplicate_chunks_are_dropped():
    """Two byte-identical chunks: only one indexed."""
    orch = _make_orchestrator()
    doc = _make_doc(["Same content here.", "Same content here."])
    added, skipped = orch._index_document(doc)
    assert added == 1
    assert skipped == 1


def test_near_duplicate_chunks_are_dropped():
    """Two chunks that differ only in whitespace/case: only one indexed."""
    orch = _make_orchestrator()
    doc = _make_doc([
        "OpenEBS scraper configuration uses Helm values for the scraper deployment.",
        "OpenEBS  Scraper   Configuration   uses Helm Values\n  for the scraper deployment.",
    ])
    added, skipped = orch._index_document(doc)
    assert added == 1, "near-duplicate should be dropped"
    assert skipped == 1


def test_distinct_chunks_both_indexed():
    """Genuinely different chunks must both be indexed."""
    orch = _make_orchestrator()
    doc = _make_doc([
        "OpenEBS scraper configuration details here.",
        "Mayastor io-engine restart procedure documentation.",
    ])
    added, skipped = orch._index_document(doc)
    assert added == 2
    assert skipped == 0


def test_fuzzy_hash_persisted_in_metadata():
    """fuzzy_hash must end up in the metadata for rebuild-time backfill."""
    orch = _make_orchestrator()
    doc = _make_doc(["Some unique content paragraph for the test."])
    orch._index_document(doc)
    add_call = orch.collection.add.call_args
    metas = add_call.kwargs["metadatas"] if add_call.kwargs else add_call[1]["metadatas"]
    assert "fuzzy_hash" in metas[0]
    assert "content_hash" in metas[0]
