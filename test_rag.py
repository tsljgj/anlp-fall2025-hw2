from rag_system.data.corpus_loader import CorpusLoader
from rag_system.retrieval.bm25_retriever import BM25Retriever
from rag_system.retrieval.dense_retriever import DenseRetriever
from rag_system.retrieval.hybrid_retriever import HybridRetriever
from rag_system.reader.llm_reader import LLMReader
from rag_system.pipeline.rag_pipeline import RAGPipeline

TOGETHER_API_KEY = "3f384ce9f4a4b21580d2fea88ef7c8dc978b6a4800596d17e3d1ad4dda1a962e"

print("Loading pre-chunked corpus...")
loader = CorpusLoader()
corpus = loader.load_texts()
print(f"Loaded {len(corpus)} chunks")

print("Building retrievers...")
bm25 = BM25Retriever(corpus)
dense = DenseRetriever(corpus, model_name="all-MiniLM-L6-v2")
hybrid = HybridRetriever(dense, bm25, alpha=0.6)

print("Setting up reader...")
reader = LLMReader(model_name="Qwen/Qwen2.5-Coder-32B-Instruct", api_key=TOGETHER_API_KEY)

print("Running RAG pipeline...")
rag = RAGPipeline(hybrid, reader, top_k=5)

question = "Who is Pittsburgh named after?"
print(f"\nQuestion: {question}")

result = rag.run(question)

print(f"\nAnswer: {result['answer']}")
print(f"\nTop context:")
print(result['contexts'][0]['text'][:200] + "...")
