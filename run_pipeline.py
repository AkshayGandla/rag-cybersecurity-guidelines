"""End-to-end RAG run: knowledge base -> embeddings -> ChromaDB -> generation -> evaluation.

Examples
--------
    python run_pipeline.py                       # default KB, Ollama (Mistral) if running, else mock generator
    python run_pipeline.py --kb large            # use data/large_kb_chunks.json (see src/build_large_kb.py)
    python run_pipeline.py --generator mock      # no LLM needed
    python run_pipeline.py --deepeval            # also run DeepEval metrics (requires an LLM judge)

Outputs are written to ./outputs (git-ignored) so the committed results/ folder is never overwritten.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation import RAGEvaluator, find_worst_cases, summary_table
from src.generation import MockGenerator, get_generator, parse_guidelines
from src.knowledge_base import get_default_chunks
from src.retrieval import ChromaRetriever, EmbeddingModel, format_context

BENCHMARK_QUERIES = [
    ("Q01", "Phishing", "How should we handle spear-phishing attempts in a mid-sized enterprise?"),
    ("Q02", "Phishing", "What email authentication standards prevent domain spoofing in phishing attacks?"),
    ("Q03", "Phishing", "Which type of MFA best protects against phishing credential theft?"),
    ("Q04", "Phishing", "What immediate steps should be taken after a user clicks a phishing link?"),
    ("Q05", "Incident Response", "What is the correct containment procedure for a ransomware incident?"),
    ("Q06", "Incident Response", "How should we structure a post-incident review after a data breach?"),
    ("Q07", "Incident Response", "What metrics should we track to measure incident response performance?"),
    ("Q08", "Vulnerability Mgmt", "How should we prioritise which CVEs to patch first?"),
    ("Q09", "Vulnerability Mgmt", "How can we detect vulnerable third-party libraries in our applications?"),
    ("Q10", "Vulnerability Mgmt", "When should we conduct penetration testing and what should it include?"),
]


def load_chunks(kb: str) -> list[dict]:
    if kb == "default":
        return get_default_chunks()
    path = Path("data/large_kb_chunks.json")
    if not path.exists():
        raise SystemExit("data/large_kb_chunks.json not found - run `python src/build_large_kb.py` first.")
    return json.loads(path.read_text(encoding="utf-8"))


def run(
    kb: str = "default",
    generator: str = "auto",
    top_k: int = 3,
    deepeval: bool = False,
    out_dir: str = "outputs",
    persist_dir: str = "./chroma_db",
    embedder: EmbeddingModel | None = None,
) -> list[dict]:
    chunks = load_chunks(kb)
    retriever = ChromaRetriever(embedder or EmbeddingModel(), persist_dir=persist_dir)
    retriever.index_chunks(chunks)
    gen = MockGenerator() if generator == "mock" else get_generator("ollama")

    records = []
    for qid, domain, query in BENCHMARK_QUERIES:
        passages = retriever.retrieve(query, top_k=top_k)
        result = gen.generate(query, format_context(passages))
        records.append(
            {
                "query_id": qid,
                "domain": domain,
                "query": query,
                "retrieved_passages": [p["text"] for p in passages],
                "passage_sources": [p["source"] for p in passages],
                "passage_scores": [p["similarity"] for p in passages],
                "response": result["output"],
                "guidelines": parse_guidelines(result["output"]),
                "model": result["model"],
                "latency_s": result["latency_s"],
            }
        )

    results = RAGEvaluator(use_deepeval=deepeval).evaluate(records)
    summary = summary_table(results)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Metric summary (mean +/- std) ===")
    for metric, s in summary.items():
        print(f"{metric:>18}: {s['mean']:.3f} +/- {s['std']:.3f}  (min {s['min']:.3f}, max {s['max']:.3f}, n={s['n']})")
    print("\n=== Lowest-faithfulness queries ===")
    for r in find_worst_cases(results, "faithfulness", 3):
        print(f"{r['query_id']} faithfulness={r['faithfulness']:.3f}  {r['query']}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kb", choices=["default", "large"], default="default")
    ap.add_argument("--generator", choices=["auto", "mock"], default="auto")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--deepeval", action="store_true")
    ap.add_argument("--out-dir", default="outputs")
    a = ap.parse_args()
    run(a.kb, a.generator, a.top_k, a.deepeval, a.out_dir)
