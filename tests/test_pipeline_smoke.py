"""Smoke test: runs the full pipeline offline with a deterministic hashing embedder and the mock generator,
so CI needs no model downloads, GPU or LLM server."""
import hashlib
import re

import numpy as np

from run_pipeline import BENCHMARK_QUERIES, run
from src.evaluation import HeuristicEvaluator
from src.knowledge_base import build_chunk_records, chunk_text, get_default_chunks


class HashingEmbedder:
    """Bag-of-words hashing embedder (L2-normalised) - stands in for sentence-transformers in tests."""

    DIM = 256

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.DIM, dtype=np.float32)
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.DIM] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed(self, texts):
        return np.vstack([self._vec(t) for t in texts])

    def embed_query(self, query):
        return self._vec(query)


def test_chunking_overlap():
    words = " ".join(f"w{i}" for i in range(700))
    chunks = chunk_text(words, chunk_size=300, overlap=50)
    assert len(chunks) == 3
    assert chunks[0].split()[-50:] == chunks[1].split()[:50]


def test_default_chunks_have_required_metadata():
    chunks = get_default_chunks()
    assert chunks and all({"chunk_id", "source", "topic", "text"} <= set(c) for c in chunks)


def test_heuristic_evaluator_ranges():
    ev = HeuristicEvaluator().evaluate_single(
        "How to stop phishing?", "Enable DMARC and MFA.", ["Phishing is reduced by DMARC and MFA."]
    )
    assert all(0.0 <= ev[k] <= 1.0 for k in ("context_relevance", "answer_relevance", "faithfulness", "composite"))


def test_end_to_end_pipeline(tmp_path):
    results = run(
        generator="mock",
        out_dir=str(tmp_path / "out"),
        persist_dir=str(tmp_path / "chroma"),
        embedder=HashingEmbedder(),
    )
    assert len(results) == len(BENCHMARK_QUERIES)
    assert all(len(r["retrieved_passages"]) == 3 for r in results)
    assert (tmp_path / "out" / "summary.json").exists()
