"""
context_compiler.py — Compress retrieved chunks into a prompt-ready context.

Pipeline:
    recall(k=20) → rank → cluster → merge → prune → prompt

Goal: fit maximum signal into LLM context window.
"""

import numpy as np
from typing import List, Callable, Optional


def _word_overlap(a: str, b: str) -> float:
    """Jaccard similarity of word sets."""
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (words × 1.3)."""
    return max(1, int(len(text.split()) * 1.3))


def _count_tokens(texts: List[str], tokenizer: Optional[Callable] = None) -> int:
    if tokenizer is not None:
        return len(tokenizer(" ".join(texts))["input_ids"])
    return sum(_estimate_tokens(t) for t in texts)


class ContextCompiler:
    """Compress retrieved BSM chunks into an LLM-ready context block.

    Args:
        similarity_threshold: Jaccard word overlap to consider chunks similar (0..1).
        max_tokens: hard token budget for the final context.
        max_chunks: hard chunk count limit.
        tokenizer: optional callable(text) → token IDs (e.g. HF tokenizer).
    """

    def __init__(self,
                 similarity_threshold: float = 0.35,
                 max_tokens: int = 768,
                 max_chunks: int = 5,
                 tokenizer: Optional[Callable] = None):
        self.similarity_threshold = similarity_threshold
        self.max_tokens = max_tokens
        self.max_chunks = max_chunks
        self.tokenizer = tokenizer

        # Stats from last compile
        self.last_stats = {}

    def compile(self,
                results: List[tuple],
                query: str = "") -> List[str]:
        """Compress raw BSM recall results into a deduplicated, ranked context list.

        Args:
            results: list of (payload_dict, hamming_dist, meta_dict)
                     from BSM.recall()
            query: query text (used for reranking if non-empty).

        Returns:
            list of chunk text strings, ordered by relevance.
        """
        if not results:
            self.last_stats = {"chunks_in": 0, "chunks_out": 0}
            return []

        # 1. Extract payloads with distances
        entries = []
        for payload, dist, meta in results:
            text = payload.get("text", str(payload))
            entries.append({"text": text, "dist": dist, "source": payload.get("source", "")})

        self.last_stats["chunks_in"] = len(entries)

        # 2. Rerank by combined Hamming distance + query word overlap
        if query:
            for e in entries:
                q_overlap = _word_overlap(e["text"], query)
                # Combined: lower is better
                # Hamming normalized by D, minus query overlap weighted
                e["query_overlap"] = q_overlap
                e["score"] = e["dist"] - q_overlap * 40.0  # weight: ~15% of D=256
            entries.sort(key=lambda e: e["score"])
        else:
            entries.sort(key=lambda e: e["dist"])
            for e in entries:
                e["query_overlap"] = 0.0
                e["score"] = float(e["dist"])

        # 3. Cluster by word overlap
        clusters = self._cluster(entries)
        self.last_stats["clusters"] = len(clusters)

        # 4. Merge: pick best representative per cluster
        merged = self._merge(clusters, entries)
        self.last_stats["merged"] = len(merged)

        # 5. Prune by token budget
        pruned = self._prune(merged)
        self.last_stats["chunks_out"] = len(pruned)
        self.last_stats["tokens"] = _count_tokens(pruned, self.tokenizer)

        return pruned

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    def _cluster(self, entries: List[dict]) -> List[List[int]]:
        """Group entry indices by pairwise word overlap."""
        N = len(entries)
        adj = [[False] * N for _ in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                sim = _word_overlap(entries[i]["text"], entries[j]["text"])
                if sim >= self.similarity_threshold:
                    adj[i][j] = adj[j][i] = True

        # Connected components (flood fill)
        visited = [False] * N
        clusters = []
        for i in range(N):
            if visited[i]:
                continue
            stack = [i]
            visited[i] = True
            comp = []
            while stack:
                v = stack.pop()
                comp.append(v)
                for u in range(N):
                    if adj[v][u] and not visited[u]:
                        visited[u] = True
                        stack.append(u)
            clusters.append(comp)
        return clusters

    def _merge(self, clusters: List[List[int]], entries: List[dict]) -> List[dict]:
        """Pick best entry per cluster: lowest score wins, tie → shortest."""
        merged = []
        for comp in clusters:
            best = min(comp, key=lambda i: (entries[i].get("score", entries[i]["dist"]),
                                            len(entries[i]["text"])))
            merged.append(entries[best])
        merged.sort(key=lambda e: e.get("score", e["dist"]))
        return merged

    def _prune(self, entries: List[dict]) -> List[str]:
        """Remove entries until within max_chunks and max_tokens."""
        out = []
        token_count = 0
        for e in entries:
            tokens = _estimate_tokens(e["text"]) if self.tokenizer is None \
                else len(self.tokenizer(e["text"])["input_ids"])
            if len(out) >= self.max_chunks:
                break
            if token_count + tokens > self.max_tokens:
                break
            out.append(e["text"])
            token_count += tokens
        return out

    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return stats from the last compile()."""
        return dict(self.last_stats)

    def __repr__(self):
        return (f"ContextCompiler(max_tokens={self.max_tokens}, "
                f"max_chunks={self.max_chunks}, "
                f"threshold={self.similarity_threshold})")
