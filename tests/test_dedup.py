"""Tests for chunk dedup (exact + fuzzy near-duplicate)."""

from pathlib import Path
from unittest.mock import MagicMock

from mcp_server.ingestion import Chunk, Document
from mcp_server.server import KnowledgeOrchestrator, _fuzzy_chunk_hash


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
