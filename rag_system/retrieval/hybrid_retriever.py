from .base_retriever import BaseRetriever
from typing import List, Dict
import numpy as np

class HybridRetriever(BaseRetriever):
    def __init__(self, dense_retriever, sparse_retriever, alpha: float = 0.5, name="hybrid"):
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.alpha = alpha
        self.name = name

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        dense_results = self.dense.retrieve(query, k * 2)
        sparse_results = self.sparse.retrieve(query, k * 2)

        scores = {}
        for doc in dense_results:
            scores[doc['text']] = self.alpha * doc['score']
        for doc in sparse_results:
            scores[doc['text']] = scores.get(doc['text'], 0) + (1 - self.alpha) * doc['score']

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [{"text": t, "score": s} for t, s in ranked]

    def get_name(self):
        return self.name
