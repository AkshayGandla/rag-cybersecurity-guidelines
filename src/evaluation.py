"""
evaluation.py
=============
RAG evaluation module using DeepEval framework.

Implements the three RAGAS-aligned dimensions required by the assignment:
  1. Context Relevance  — are retrieved passages relevant to the query?
  2. Answer Relevance   — does the generated guideline answer the query?
  3. Faithfulness       — is the answer grounded in the retrieved context?

Architecture:
  - Primary: DeepEval ContextualRelevancyMetric, AnswerRelevancyMetric,
             FaithfulnessMetric (reference-free, no ground truth needed).
  - Fallback: Custom heuristic scorers based on cosine similarity and
              token overlap — used when DeepEval is unavailable or for
              ablation comparison.

Why DeepEval over raw RAGAS?
  DeepEval wraps RAGAS-style metrics in a clean pytest-compatible API,
  supports custom LLM judges (including local Ollama), and produces
  structured score reports. The assignment explicitly permits alternative
  frameworks; DeepEval was selected for its active maintenance and
  industry adoption (used by teams at Stripe, Shopify, etc.).

Reference: https://docs.confident-ai.com/docs/getting-started
"""

import json
import re
import math
from typing import List, Dict, Tuple, Optional

import numpy as np


# ---------------------------------------------------------------------------
# DeepEval integration
# ---------------------------------------------------------------------------

def _try_import_deepeval():
    try:
        from deepeval import evaluate
        from deepeval.metrics import (
            ContextualRelevancyMetric,
            AnswerRelevancyMetric,
            FaithfulnessMetric,
        )
        from deepeval.test_case import LLMTestCase
        return evaluate, ContextualRelevancyMetric, AnswerRelevancyMetric, FaithfulnessMetric, LLMTestCase
    except ImportError:
        return None


class DeepEvalEvaluator:
    """
    Runs DeepEval's three core RAG metrics on a set of test cases.

    Metrics:
      - ContextualRelevancyMetric: scores how relevant each retrieved
        chunk is to the query (average across chunks).
      - AnswerRelevancyMetric: scores whether the response addresses
        the query's intent.
      - FaithfulnessMetric: scores whether claims in the response are
        entailed by the retrieved context (hallucination detection).

    All metrics are reference-free (no gold-standard answer required).
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        threshold: float = 0.5,
        use_local: bool = True,
    ):
        """
        Parameters
        ----------
        model      : LLM judge model name. For local eval, pass "ollama/mistral".
        threshold  : minimum passing score (0–1).
        use_local  : if True, attempt to configure Ollama as judge.
        """
        imports = _try_import_deepeval()
        if imports is None:
            raise ImportError(
                "DeepEval not installed. Run: pip install deepeval"
            )
        (
            self.evaluate,
            self.ContextualRelevancyMetric,
            self.AnswerRelevancyMetric,
            self.FaithfulnessMetric,
            self.LLMTestCase,
        ) = imports

        self.threshold = threshold
        self.model = model
        self.use_local = use_local

    def _make_metrics(self):
        kwargs = {"threshold": self.threshold, "verbose_mode": False}
        # DeepEval ≥0.21 accepts model= kwarg
        try:
            return [
                self.ContextualRelevancyMetric(**kwargs),
                self.AnswerRelevancyMetric(**kwargs),
                self.FaithfulnessMetric(**kwargs),
            ]
        except TypeError:
            # Older DeepEval versions
            return [
                self.ContextualRelevancyMetric(threshold=self.threshold),
                self.AnswerRelevancyMetric(threshold=self.threshold),
                self.FaithfulnessMetric(threshold=self.threshold),
            ]

    def evaluate_single(
        self,
        query: str,
        response: str,
        retrieved_passages: List[str],
    ) -> Dict:
        """
        Evaluate a single (query, response, context) triple.

        Returns dict with scores for each metric dimension.
        """
        test_case = self.LLMTestCase(
            input=query,
            actual_output=response,
            retrieval_context=retrieved_passages,
        )
        metrics = self._make_metrics()
        scores = {}
        for metric in metrics:
            try:
                metric.measure(test_case)
                metric_name = type(metric).__name__.replace("Metric", "").lower()
                # Normalise name
                if "contextual" in metric_name:
                    key = "context_relevance"
                elif "answer" in metric_name:
                    key = "answer_relevance"
                elif "faithful" in metric_name:
                    key = "faithfulness"
                else:
                    key = metric_name
                scores[key] = round(float(metric.score), 4)
                scores[f"{key}_reason"] = getattr(metric, "reason", "")
                scores[f"{key}_pass"] = metric.is_successful()
            except Exception as e:
                # Graceful degradation: log and continue
                print(f"[DeepEval] Warning: {type(metric).__name__} failed: {e}")
                scores[key] = -1.0
        return scores

    def evaluate_dataset(self, records: List[Dict]) -> List[Dict]:
        """
        Evaluate a list of records. Each record must have:
          'query', 'response', 'retrieved_passages' (list of str)

        Returns records enriched with metric scores.
        """
        results = []
        for i, rec in enumerate(records, 1):
            print(f"[DeepEval] Evaluating record {i}/{len(records)}: {rec['query'][:60]}…")
            scores = self.evaluate_single(
                query=rec["query"],
                response=rec["response"],
                retrieved_passages=rec["retrieved_passages"],
            )
            results.append({**rec, **scores})
        return results


# ---------------------------------------------------------------------------
# Heuristic fallback evaluator (no LLM judge needed)
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> set:
    """Lowercase word tokens, removing punctuation."""
    return set(re.findall(r"\b\w+\b", text.lower()))


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokenise(a), _tokenise(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def cosine_sim_tokens(a: str, b: str) -> float:
    """Bag-of-words cosine similarity between two strings."""
    all_words = list(_tokenise(a) | _tokenise(b))
    if not all_words:
        return 0.0
    def vec(text):
        tokens = _tokenise(text)
        return np.array([1.0 if w in tokens else 0.0 for w in all_words])
    va, vb = vec(a), vec(b)
    denom = (np.linalg.norm(va) * np.linalg.norm(vb))
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


class HeuristicEvaluator:
    """
    Reference-free heuristic RAG evaluator.

    Used as:
      (a) A fallback when DeepEval is unavailable.
      (b) A comparison baseline in the evaluation analysis section.

    Metric definitions:
      context_relevance:
        Average cosine similarity (token BoW) between query and each
        retrieved passage. Higher = passages are more on-topic.

      answer_relevance:
        Cosine similarity between the query and the generated response.
        Measures whether the response stays on-topic.

      faithfulness:
        Proportion of sentences in the response that share at least
        one non-stop-word token with the retrieved context.
        Approximates entailment without an LLM judge.
    """

    STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "shall", "to", "of", "in",
        "for", "on", "with", "at", "by", "from", "that", "this", "it",
        "as", "or", "and", "but", "not", "if", "into", "through",
        "during", "before", "after", "above", "below", "between",
        "each", "both", "few", "more", "most", "other", "some", "such",
        "than", "then", "when", "where", "which", "while", "who",
        "whom", "how", "all", "any", "can", "its", "also", "their",
        "your", "our", "his", "her", "we", "they", "you", "i",
    }

    def _content_tokens(self, text: str) -> set:
        return _tokenise(text) - self.STOPWORDS

    def context_relevance(self, query: str, passages: List[str]) -> float:
        """Average cosine sim between query and each retrieved passage."""
        if not passages:
            return 0.0
        sims = [cosine_sim_tokens(query, p) for p in passages]
        return round(float(np.mean(sims)), 4)

    def answer_relevance(self, query: str, response: str) -> float:
        """Cosine sim between query tokens and response tokens."""
        return round(cosine_sim_tokens(query, response), 4)

    def faithfulness(self, response: str, passages: List[str]) -> float:
        """
        Fraction of response sentences containing ≥1 content token
        that also appears in the retrieved context.
        """
        context_tokens = set()
        for p in passages:
            context_tokens |= self._content_tokens(p)

        sentences = [s.strip() for s in re.split(r"[.!?;]", response) if s.strip()]
        if not sentences:
            return 0.0

        supported = 0
        for sent in sentences:
            sent_tokens = self._content_tokens(sent)
            if sent_tokens & context_tokens:
                supported += 1
        return round(supported / len(sentences), 4)

    def evaluate_single(
        self,
        query: str,
        response: str,
        retrieved_passages: List[str],
    ) -> Dict:
        cr  = self.context_relevance(query, retrieved_passages)
        ar  = self.answer_relevance(query, response)
        fth = self.faithfulness(response, retrieved_passages)
        return {
            "context_relevance": cr,
            "answer_relevance":  ar,
            "faithfulness":      fth,
            "composite":         round((cr + ar + fth) / 3, 4),
        }

    def evaluate_dataset(self, records: List[Dict]) -> List[Dict]:
        results = []
        for rec in records:
            scores = self.evaluate_single(
                query=rec["query"],
                response=rec["response"],
                retrieved_passages=rec["retrieved_passages"],
            )
            results.append({**rec, **scores})
        return results


# ---------------------------------------------------------------------------
# Unified evaluator — tries DeepEval first, falls back to heuristic
# ---------------------------------------------------------------------------

class RAGEvaluator:
    """
    Unified evaluator that tries DeepEval first and falls back gracefully.
    Always runs the heuristic evaluator for comparison/ablation.
    """

    def __init__(self, use_deepeval: bool = True):
        self.heuristic = HeuristicEvaluator()
        self.deepeval_evaluator = None
        if use_deepeval:
            try:
                self.deepeval_evaluator = DeepEvalEvaluator()
                print("[RAGEvaluator] DeepEval loaded successfully.")
            except (ImportError, Exception) as e:
                print(f"[RAGEvaluator] DeepEval unavailable ({e}). Using heuristic only.")

    def evaluate(self, records: List[Dict]) -> List[Dict]:
        """
        Evaluate all records.
        Each record must have: query, response, retrieved_passages (list[str]).
        Returns records enriched with heuristic scores and (if available)
        DeepEval scores.
        """
        print(f"[RAGEvaluator] Running heuristic evaluation on {len(records)} records…")
        results = self.heuristic.evaluate_dataset(records)

        if self.deepeval_evaluator:
            print("[RAGEvaluator] Running DeepEval evaluation…")
            try:
                deepeval_results = self.deepeval_evaluator.evaluate_dataset(records)
                # Merge DeepEval scores into results under 'de_' prefix
                for i, dr in enumerate(deepeval_results):
                    for k in ["context_relevance", "answer_relevance", "faithfulness"]:
                        results[i][f"de_{k}"] = dr.get(k, -1.0)
            except Exception as e:
                print(f"[RAGEvaluator] DeepEval batch evaluation failed: {e}")
        return results


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def summary_table(results: List[Dict]) -> Dict:
    """Compute mean, std, min, max for each metric across all results."""
    metrics = ["context_relevance", "answer_relevance", "faithfulness", "composite"]
    summary = {}
    for m in metrics:
        vals = [r[m] for r in results if m in r and r[m] >= 0]
        if vals:
            summary[m] = {
                "mean": round(float(np.mean(vals)), 4),
                "std":  round(float(np.std(vals)), 4),
                "min":  round(float(np.min(vals)), 4),
                "max":  round(float(np.max(vals)), 4),
                "n":    len(vals),
            }
    return summary


def find_worst_cases(results: List[Dict], metric: str = "faithfulness", n: int = 3) -> List[Dict]:
    """Return the n records with lowest scores for *metric*."""
    valid = [r for r in results if metric in r and r[metric] >= 0]
    return sorted(valid, key=lambda x: x[metric])[:n]
