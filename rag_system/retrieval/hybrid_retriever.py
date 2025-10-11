import numpy as np
from typing import List, Dict
from sentence_transformers import CrossEncoder
from .base_retriever import BaseRetriever


def normalize(scores):
    """Normalize a list/array of scores to [0, 1] range."""
    scores = np.array(scores, dtype=float)
    if len(scores) == 0:
        return scores
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)


class HybridRetriever(BaseRetriever):
    def __init__(
        self,
        dense_retriever,
        sparse_retriever,
        alpha: float = 0.5,
        use_reranker: bool = False,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        name="hybrid",
    ):
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.alpha = alpha
        self.name = name
        self.use_reranker = use_reranker

        if use_reranker:
            print(f"Loading reranker: {reranker_model}")
            self.reranker = CrossEncoder(reranker_model)

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        dense_results = self.dense.retrieve(query, k * 2)
        sparse_results = self.sparse.retrieve(query, k * 2)

        dense_scores = {d["text"]: d["score"] for d in dense_results}
        sparse_scores = {s["text"]: s["score"] for s in sparse_results}

        all_dense = list(dense_scores.values())
        all_sparse = list(sparse_scores.values())

        dense_norm = normalize(all_dense)
        sparse_norm = normalize(all_sparse)

        for i, key in enumerate(dense_scores.keys()):
            dense_scores[key] = dense_norm[i]
        for i, key in enumerate(sparse_scores.keys()):
            sparse_scores[key] = sparse_norm[i]

        # === Adaptive alpha based on query length ===
        if len(query.split()) < 5:
            alpha = 0.3  # short query → lexical match dominates
        else:
            alpha = 0.7  # long query → semantic match dominates
        # ============================================

        combined_scores = {}
        all_texts = set(dense_scores.keys()) | set(sparse_scores.keys())

        for text in all_texts:
            d = dense_scores.get(text, 0.0)
            s = sparse_scores.get(text, 0.0)
            combined_scores[text] = alpha * d + (1 - alpha) * s

            ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            top_docs = [{"text": t, "score": s} for t, s in ranked[: k * 2]]

        if self.use_reranker:
            pairs = [(query, doc["text"]) for doc in top_docs]
            rerank_scores = self.reranker.predict(pairs)
            reranked = sorted(
                zip(top_docs, rerank_scores),
                key=lambda x: x[1],
                reverse=True,
            )
            top_docs = [{"text": d["text"], "score": float(s)} for d, s in reranked[:k]]

        return top_docs

    def get_name(self):
        return self.name
