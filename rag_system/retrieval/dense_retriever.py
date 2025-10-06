from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List
from .base_retriever import BaseRetriever

class DenseRetriever(BaseRetriever):
    def __init__(self, corpus: List[str], model_name="all-MiniLM-L6-v2", name="dense"):
        self.name = name
        self.model = SentenceTransformer(model_name)
        self.corpus = corpus
        self.embeddings = self.model.encode(corpus, convert_to_numpy=True, show_progress_bar=True)
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def retrieve(self, query: str, k: int = 5):
        q_emb = self.model.encode([query], convert_to_numpy=True)
        scores, idxs = self.index.search(q_emb, k)
        return [{"text": self.corpus[i], "score": float(scores[0][j])} for j, i in enumerate(idxs[0])]

    def get_name(self):
        return self.name
