"""
Test script for DocumentChunker
Tests rolling window chunking and other strategies on real knowledge base data.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from chunker import DocumentChunker, Chunk


def load_knowledge_base(filepath: str, max_docs: int = None) -> list:
    """Load documents from JSONL knowledge base."""
    documents = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if max_docs and idx >= max_docs:
                break
            doc = json.loads(line)
            documents.append(doc)
    return documents


def analyze_documents(documents: list):
    """Analyze document statistics."""
    print(f"\n{'='*70}")
    print("DOCUMENT ANALYSIS")
    print(f"{'='*70}")

    total_docs = len(documents)
    content_lengths = [len(doc.get('content', '')) for doc in documents]
    token_counts = [len(doc.get('content', '').split()) for doc in documents]

    print(f"\nTotal documents: {total_docs}")
    print(f"\nContent Length (characters):")
    print(f"  Min: {min(content_lengths):,}")
    print(f"  Max: {max(content_lengths):,}")
    print(f"  Average: {sum(content_lengths) // len(content_lengths):,}")

    print(f"\nToken Count (whitespace):")
    print(f"  Min: {min(token_counts):,}")
    print(f"  Max: {max(token_counts):,}")
    print(f"  Average: {sum(token_counts) // len(token_counts):,}")

    # Category breakdown
    categories = defaultdict(int)
    for doc in documents:
        categories[doc.get('source_category', 'unknown')] += 1

    print(f"\nDocuments by Category:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")


def test_chunking_strategy(
    chunker: DocumentChunker,
    documents: list,
    strategy: str,
    strategy_name: str
):
    """Test a specific chunking strategy."""
    print(f"\n{'='*70}")
    print(f"TESTING: {strategy_name}")
    print(f"{'='*70}")
    print(f"Strategy: {strategy}")
    print(f"Chunk size: {chunker.chunk_size} tokens")
    print(f"Overlap: {chunker.overlap} tokens")

    # Chunk all documents
    all_chunks = chunker.chunk_documents(documents, strategy=strategy)

    # Statistics
    chunk_texts = chunker.chunks_to_texts(all_chunks)
    chunk_token_counts = [chunk.token_count for chunk in all_chunks]

    print(f"\n📊 RESULTS:")
    print(f"  Total chunks created: {len(all_chunks):,}")
    print(f"  Chunks per document (avg): {len(all_chunks) / len(documents):.2f}")

    print(f"\n  Chunk Token Counts:")
    print(f"    Min: {min(chunk_token_counts)}")
    print(f"    Max: {max(chunk_token_counts)}")
    print(f"    Average: {sum(chunk_token_counts) // len(chunk_token_counts)}")

    print(f"\n  Chunk Character Lengths:")
    chunk_char_lengths = [len(text) for text in chunk_texts]
    print(f"    Min: {min(chunk_char_lengths)}")
    print(f"    Max: {max(chunk_char_lengths)}")
    print(f"    Average: {sum(chunk_char_lengths) // len(chunk_char_lengths)}")

    # Show sample chunks
    print(f"\n📄 SAMPLE CHUNKS (first 3):")
    for i, chunk in enumerate(all_chunks[:3]):
        print(f"\n  --- Chunk {i+1} ---")
        print(f"  ID: {chunk.chunk_id}")
        print(f"  Source: {chunk.source_title[:60]}...")
        print(f"  Category: {chunk.source_category}")
        print(f"  Tokens: {chunk.token_count}")
        print(f"  Text preview: {chunk.text[:200]}...")

    return all_chunks, chunk_texts


def test_edge_cases(chunker: DocumentChunker):
    """Test edge cases."""
    print(f"\n{'='*70}")
    print("EDGE CASE TESTING")
    print(f"{'='*70}")

    # Empty document
    empty_doc = {"content": "", "title": "Empty", "url": "test", "source_category": "test"}
    chunks = chunker.chunk_document(empty_doc)
    print(f"\n✓ Empty document: {len(chunks)} chunks (expected: 0)")

    # Very short document (< chunk size)
    short_doc = {
        "content": "This is a short document with only a few words.",
        "title": "Short",
        "url": "test",
        "source_category": "test"
    }
    chunks = chunker.chunk_document(short_doc)
    print(f"✓ Short document: {len(chunks)} chunks (expected: 1)")
    if chunks:
        print(f"  Token count: {chunks[0].token_count}")

    # Very long single sentence
    long_sentence = {
        "content": " ".join(["word"] * 2000),
        "title": "Long Sentence",
        "url": "test",
        "source_category": "test"
    }
    chunks = chunker.chunk_document(long_sentence, strategy="rolling")
    print(f"✓ Long sentence (2000 words): {len(chunks)} chunks")
    print(f"  Chunks created: {len(chunks)}")

    # Document with many paragraphs
    para_doc = {
        "content": "\n\n".join([f"Paragraph {i}. " + " ".join(["word"] * 50) for i in range(20)]),
        "title": "Many Paragraphs",
        "url": "test",
        "source_category": "test"
    }
    chunks = chunker.chunk_document(para_doc, strategy="paragraph")
    print(f"✓ Document with 20 paragraphs: {len(chunks)} chunks")


def compare_strategies(documents: list, chunk_size: int = 512, overlap: int = 128):
    """Compare all chunking strategies side-by-side."""
    print(f"\n{'='*70}")
    print("STRATEGY COMPARISON")
    print(f"{'='*70}")

    strategies = [
        ("rolling", "Rolling Window (Sliding Window)"),
        ("sentence", "Sentence-Aware"),
        ("paragraph", "Paragraph-Based"),
        ("semantic", "Semantic (Advanced Sentence-Aware)")
    ]

    results = {}

    for strategy, name in strategies:
        chunker = DocumentChunker(chunk_size=chunk_size, overlap=overlap)
        chunks, texts = test_chunking_strategy(chunker, documents, strategy, name)
        results[strategy] = {
            "chunks": chunks,
            "texts": texts,
            "count": len(chunks),
            "avg_tokens": sum(c.token_count for c in chunks) / len(chunks) if chunks else 0
        }

    # Comparison table
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Strategy':<30} {'Chunks':<10} {'Avg Tokens':<12} {'Chunks/Doc'}")
    print("-" * 70)

    for strategy, name in strategies:
        res = results[strategy]
        chunks_per_doc = res['count'] / len(documents) if documents else 0
        print(f"{name:<30} {res['count']:<10} {res['avg_tokens']:<12.1f} {chunks_per_doc:.2f}")


def test_parameter_variations(documents: list):
    """Test different chunk size and overlap parameters."""
    print(f"\n{'='*70}")
    print("PARAMETER VARIATION TESTING")
    print(f"{'='*70}")

    configs = [
        (256, 64),   # Small chunks, small overlap
        (512, 128),  # Medium chunks, medium overlap
        (512, 256),  # Medium chunks, large overlap
        (1024, 128), # Large chunks, small overlap
    ]

    print(f"\n{'Chunk Size':<12} {'Overlap':<10} {'Total Chunks':<15} {'Avg Tokens':<12}")
    print("-" * 70)

    for chunk_size, overlap in configs:
        chunker = DocumentChunker(chunk_size=chunk_size, overlap=overlap)
        chunks = chunker.chunk_documents(documents, strategy="rolling")

        avg_tokens = sum(c.token_count for c in chunks) / len(chunks) if chunks else 0

        print(f"{chunk_size:<12} {overlap:<10} {len(chunks):<15} {avg_tokens:<12.1f}")


def save_sample_chunks(chunks: list, output_file: str, num_samples: int = 10):
    """Save sample chunks to file for inspection."""
    print(f"\n💾 Saving {num_samples} sample chunks to {output_file}...")

    with open(output_file, 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(chunks[:num_samples]):
            f.write(f"{'='*70}\n")
            f.write(f"CHUNK {i+1}\n")
            f.write(f"{'='*70}\n")
            f.write(f"ID: {chunk.chunk_id}\n")
            f.write(f"Source Title: {chunk.source_title}\n")
            f.write(f"Source URL: {chunk.source_url}\n")
            f.write(f"Category: {chunk.source_category}\n")
            f.write(f"Chunk Index: {chunk.chunk_index}\n")
            f.write(f"Token Count: {chunk.token_count}\n")
            f.write(f"Character Range: {chunk.start_char}-{chunk.end_char}\n")
            f.write(f"\nCONTENT:\n")
            f.write(f"{chunk.text}\n\n")

    print(f"✓ Sample chunks saved!")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("DOCUMENT CHUNKER TEST SUITE")
    print("="*70)

    # Path to knowledge base
    kb_path = Path(__file__).parent.parent.parent / "data_collection" / "data" / "pittsburgh_cmu_knowledge_base.jsonl"

    if not kb_path.exists():
        print(f"\n❌ Error: Knowledge base not found at {kb_path}")
        print("Please ensure the data collection has been completed.")
        sys.exit(1)

    print(f"\n📂 Loading knowledge base from: {kb_path}")

    # Load subset for testing
    print("Loading first 20 documents for testing...")
    documents = load_knowledge_base(str(kb_path), max_docs=20)

    if not documents:
        print("❌ Error: No documents loaded!")
        sys.exit(1)

    print(f"✓ Loaded {len(documents)} documents")

    # Run tests
    analyze_documents(documents)

    # Test edge cases
    chunker = DocumentChunker(chunk_size=512, overlap=128)
    test_edge_cases(chunker)

    # Compare strategies
    compare_strategies(documents, chunk_size=512, overlap=128)

    # Test parameter variations
    test_parameter_variations(documents)

    # Create final chunks and save samples
    print(f"\n{'='*70}")
    print("FINAL CHUNKING (Rolling Window)")
    print(f"{'='*70}")

    chunker = DocumentChunker(chunk_size=512, overlap=128)
    final_chunks = chunker.chunk_documents(documents, strategy="rolling")

    output_file = Path(__file__).parent / "sample_chunks_output.txt"
    save_sample_chunks(final_chunks, str(output_file), num_samples=10)

    # Summary
    print(f"\n{'='*70}")
    print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
    print(f"{'='*70}")
    print(f"\nFinal Statistics:")
    print(f"  Documents processed: {len(documents)}")
    print(f"  Total chunks created: {len(final_chunks)}")
    print(f"  Average chunks per document: {len(final_chunks) / len(documents):.2f}")
    print(f"  Sample output saved to: {output_file}")

    print("\n💡 Next steps:")
    print("  1. Review sample chunks in sample_chunks_output.txt")
    print("  2. Run on full knowledge base (530 documents)")
    print("  3. Integrate with corpus_builder.py")
    print("  4. Test with BM25/Dense retrievers")


if __name__ == "__main__":
    main()
