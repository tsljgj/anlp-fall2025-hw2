from rank_bm25 import BM25Okapi
from .base_retriever import BaseRetriever
import numpy as np
from typing import List

class BM25Retriever(BaseRetriever):
    def __init__(self, corpus: List[str], name: str = "bm25"):
        self.name = name
        self.corpus = corpus
        self.tokenized_corpus = [doc.split() for doc in corpus]
        self.model = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query: str, k: int = 5):
        tokenized_query = query.split()
        scores = self.model.get_scores(tokenized_query)
        topk_idx = np.argsort(scores)[::-1][:k]
        return [{"text": self.corpus[i], "score": float(scores[i])} for i in topk_idx]

    def get_name(self):
        return self.name
