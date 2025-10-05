from rag_system.retrieval.bm25_retriever import BM25Retriever
from rag_system.retrieval.dense_retriever import DenseRetriever
from rag_system.retrieval.hybrid_retriever import HybridRetriever
from rag_system.reader.qa_reader import QAReader
from rag_system.pipeline.rag_pipeline import RAGPipeline


corpus = [
    "Pittsburgh was named after William Pitt.",
    "The first ICML conference was held in Pittsburgh in 1980."
]

bm25 = BM25Retriever(corpus)
dense = DenseRetriever(corpus)
hybrid = HybridRetriever(dense, bm25, alpha=0.6)
reader = QAReader()

rag = RAGPipeline(hybrid, reader)
result = rag.run("Who is Pittsburgh named after?")
print(result)
