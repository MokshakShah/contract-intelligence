"""Pluggable BM25, dense, hybrid, and optional cross-encoder retrieval."""

import re
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from .chunking import Chunk
from .abstention import normalized_scores


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class Retriever:
    def __init__(self, chunks: list[Chunk], config: dict[str, Any]):
        self.chunks = chunks
        self.config = config
        self.method = config.get("method", "bm25")
        self.top_k = int(config.get("top_k", 8))
        self.max_per_doc = config.get("max_chunks_per_doc")
        self.bm25 = BM25Okapi([_tokens(chunk.text) for chunk in chunks])
        self.dense_model = None
        self.embeddings = None
        if self.method in {"dense", "hybrid"} or config.get("rerank"):
            self._load_dense()

    def _load_dense(self) -> None:
        from sentence_transformers import SentenceTransformer

        model_name = self.config.get("dense_model", "BAAI/bge-small-en-v1.5")
        cache_dir = Path(self.config.get("cache_dir", ".cache"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.dense_model = SentenceTransformer(model_name, cache_folder=str(cache_dir))
        cache_file = cache_dir / "chunk_embeddings.npy"
        if cache_file.exists() and len(np.load(cache_file, mmap_mode="r")) == len(self.chunks):
            self.embeddings = np.load(cache_file)
        else:
            self.embeddings = self.dense_model.encode(
                [chunk.text for chunk in self.chunks],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            np.save(cache_file, self.embeddings)

    def _candidates(self, query: str) -> list[tuple[int, float]]:
        scores: dict[int, float] = {}
        if self.method in {"bm25", "hybrid"}:
            for i, score in enumerate(self.bm25.get_scores(_tokens(query))):
                scores[i] = float(score)
        if self.method in {"dense", "hybrid"}:
            query_embedding = self.dense_model.encode([query], normalize_embeddings=True)[0]
            dense_scores = self.embeddings @ query_embedding
            if self.method == "dense":
                scores = {i: float(score) for i, score in enumerate(dense_scores)}
            else:
                ranks_bm = {i: rank for rank, (i, _) in enumerate(sorted(scores.items(), key=lambda x: x[1], reverse=True), 1)}
                ranks_dense = {i: rank for rank, i in enumerate(np.argsort(-dense_scores), 1)}
                rrf_k = float(self.config.get("rrf_k", 60))
                scores = {i: 1 / (rrf_k + ranks_bm[i]) + 1 / (rrf_k + ranks_dense[i]) for i in scores}
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        limit = top_k or self.top_k
        candidates = self._candidates(query)
        if self.config.get("rerank"):
            from sentence_transformers import CrossEncoder

            n = int(self.config.get("rerank_top_n", max(limit, 32)))
            model = CrossEncoder(self.config.get("rerank_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
            pairs = [(query, self.chunks[i].text) for i, _ in candidates[:n]]
            reranked = model.predict(pairs)
            candidates = sorted(zip((i for i, _ in candidates[:n]), reranked), key=lambda x: x[1], reverse=True)
        normalized = normalized_scores([float(score) for _, score in candidates])
        results = []
        doc_counts: dict[str, int] = {}
        for position, (index, score) in enumerate(candidates):
            chunk = self.chunks[index]
            if self.max_per_doc is not None and doc_counts.get(chunk.doc_path, 0) >= int(self.max_per_doc):
                continue
            doc_counts[chunk.doc_path] = doc_counts.get(chunk.doc_path, 0) + 1
            results.append({
                **chunk.__dict__,
                "score": float(score),
                "normalized_score": normalized[position],
            })
            if len(results) >= limit:
                break
        return results
