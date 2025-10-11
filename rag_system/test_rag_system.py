"""
End-to-End RAG System Test
Tests the complete RAG pipeline with real knowledge base and 5 test questions.
"""

import json
import sys
import time
from pathlib import Path
from typing import List, Dict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_system.data.chunker import DocumentChunker
from rag_system.retrieval.bm25_retriever import BM25Retriever
from rag_system.retrieval.dense_retriever import DenseRetriever
from rag_system.retrieval.hybrid_retriever import HybridRetriever
from rag_system.reader.llm_reader import LLMReader
from rag_system.pipeline.rag_pipeline import RAGPipeline


def load_knowledge_base(filepath: str) -> List[Dict]:
    """Load all documents from knowledge base."""
    print(f"📂 Loading knowledge base from: {filepath}")
    documents = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            documents.append(doc)
    print(f"✓ Loaded {len(documents)} documents")
    return documents


def chunk_knowledge_base(documents: List[Dict], chunk_size: int = 512, overlap: int = 128) -> tuple:
    """Chunk all documents and return chunks."""
    print(f"\n📝 Chunking documents (chunk_size={chunk_size}, overlap={overlap})...")

    chunker = DocumentChunker(chunk_size=chunk_size, overlap=overlap)
    chunks = chunker.chunk_documents(documents, strategy="rolling")
    chunk_texts = chunker.chunks_to_texts(chunks)

    print(f"✓ Created {len(chunks)} chunks")
    print(f"  Average tokens per chunk: {sum(c.token_count for c in chunks) / len(chunks):.1f}")
    print(f"  Average chunks per document: {len(chunks) / len(documents):.2f}")

    return chunks, chunk_texts


def build_retrievers(chunk_texts: List[str]):
    """Build all retrievers with the chunked corpus."""
    print(f"\n🔍 Building retrievers...")

    print("  Building BM25 retriever...")
    bm25 = BM25Retriever(chunk_texts, name="BM25")

    print("  Building Dense retriever (this may take a few minutes)...")
    dense = DenseRetriever(chunk_texts, model_name="all-MiniLM-L6-v2", name="Dense")

    print("  Building Hybrid retriever...")
    hybrid = HybridRetriever(dense, bm25, alpha=0.6, name="Hybrid")

    print("✓ All retrievers built successfully")

    return bm25, dense, hybrid


def setup_reader(api_key: str = None):
    """Setup the LLM reader."""
    print(f"\n🤖 Setting up LLM reader...")

    # Try to use the API key from test_rag.py if not provided
    if not api_key:
        try:
            test_rag_path = Path(__file__).parent.parent / "test_rag.py"
            with open(test_rag_path, 'r') as f:
                content = f.read()
                # Extract API key (simple regex)
                import re
                match = re.search(r'TOGETHER_API_KEY\s*=\s*["\']([^"\']+)["\']', content)
                if match:
                    api_key = match.group(1)
                    print("✓ Found API key from test_rag.py")
        except:
            pass

    if not api_key:
        print("⚠️ Warning: No API key found. Using placeholder.")
        api_key = "YOUR_API_KEY_HERE"

    reader = LLMReader(model_name="Qwen/Qwen2.5-Coder-32B-Instruct", api_key=api_key)
    print("✓ LLM reader ready")

    return reader


def format_retrieved_context(contexts: List[Dict], max_display: int = 3) -> str:
    """Format retrieved contexts for display."""
    output = []
    for i, ctx in enumerate(contexts[:max_display], 1):
        output.append(f"  [{i}] Score: {ctx['score']:.4f}")
        output.append(f"      Text: {ctx['text'][:150]}...")
    return "\n".join(output)


def run_rag_test(retriever, reader, questions: List[str], retriever_name: str):
    """Run RAG pipeline on test questions."""
    print(f"\n{'='*80}")
    print(f"Testing with {retriever_name} Retriever")
    print(f"{'='*80}")

    pipeline = RAGPipeline(retriever, reader, top_k=5)

    results = []
    for i, question in enumerate(questions, 1):
        print(f"\n{'─'*80}")
        print(f"Question {i}: {question}")
        print(f"{'─'*80}")

        try:
            start_time = time.time()
            result = pipeline.run(question)
            elapsed = time.time() - start_time

            print(f"\n📊 Retrieved Contexts (top 3 of 5):")
            print(format_retrieved_context(result['contexts']))

            print(f"\n💡 Answer: {result['answer']}")
            print(f"\n⏱️  Time: {elapsed:.2f}s")

            results.append({
                "question": question,
                "answer": result['answer'],
                "contexts": result['contexts'],
                "time": elapsed
            })

        except Exception as e:
            print(f"\n❌ Error: {e}")
            results.append({
                "question": question,
                "answer": f"Error: {e}",
                "contexts": [],
                "time": 0
            })

    return results


def save_results(all_results: Dict[str, List], output_file: str):
    """Save test results to JSON file."""
    print(f"\n💾 Saving results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print("✓ Results saved")


def main():
    """Run the complete RAG system test."""
    print("\n" + "="*80)
    print("END-TO-END RAG SYSTEM TEST")
    print("="*80)

    # Configuration
    kb_path = Path(__file__).parent.parent / "data_collection" / "data" / "pittsburgh_cmu_knowledge_base.jsonl"
    chunk_size = 512
    overlap = 128

    # Test questions about Pittsburgh and CMU
    test_questions = [
        "Who is Pittsburgh named after?",
        "When was Carnegie Mellon University founded?",
        "What is the name of the annual pickle festival held in Pittsburgh?",
        "What are the three major professional sports teams in Pittsburgh?",
        "What is Spring Carnival at CMU?"
    ]

    print("\n📋 Test Questions:")
    for i, q in enumerate(test_questions, 1):
        print(f"  {i}. {q}")

    # Check if knowledge base exists
    if not kb_path.exists():
        print(f"\n❌ Error: Knowledge base not found at {kb_path}")
        print("Please ensure data collection has been completed.")
        sys.exit(1)

    # Step 1: Load knowledge base
    documents = load_knowledge_base(str(kb_path))

    # Step 2: Chunk documents
    chunks, chunk_texts = chunk_knowledge_base(documents, chunk_size, overlap)

    # Step 3: Build retrievers
    bm25, dense, hybrid = build_retrievers(chunk_texts)

    # Step 4: Setup reader
    reader = setup_reader()

    # Step 5: Run tests with different retrievers
    all_results = {}

    # Test with each retriever
    retrievers_to_test = [
        ("BM25", bm25),
        ("Dense", dense),
        ("Hybrid", hybrid)
    ]

    for name, retriever in retrievers_to_test:
        results = run_rag_test(retriever, reader, test_questions, name)
        all_results[name] = results

    # Step 6: Save results
    output_file = Path(__file__).parent / "test_results.json"
    save_results(all_results, str(output_file))

    # Summary
    print(f"\n{'='*80}")
    print("✅ TEST COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}")
    print(f"\n📊 Summary:")
    print(f"  Documents processed: {len(documents)}")
    print(f"  Total chunks created: {len(chunks)}")
    print(f"  Questions tested: {len(test_questions)}")
    print(f"  Retrievers tested: {len(retrievers_to_test)}")
    print(f"  Results saved to: {output_file}")
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
