import json
import pickle
from pathlib import Path
from typing import List, Dict
from .chunker import DocumentChunker


class CorpusBuilder:
    def __init__(self, kb_path: str, chunk_size: int = 512, overlap: int = 128, strategy: str = "rolling"):
        """Initialize corpus builder with chunking configuration."""
        self.kb_path = Path(kb_path)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.strategy = strategy
        self.documents = []
        self.chunks = []
        self.chunk_texts = []

    def load_knowledge_base(self) -> List[Dict]:
        """Load all documents from knowledge base JSONL file."""
        documents = []
        with open(self.kb_path, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line)
                documents.append(doc)
        self.documents = documents
        return documents

    def build_corpus(self) -> tuple:
        """Chunk all documents and prepare corpus."""
        if not self.documents:
            raise ValueError("No documents loaded. Call load_knowledge_base() first.")

        chunker = DocumentChunker(
            chunk_size=self.chunk_size,
            overlap=self.overlap
        )

        self.chunks = chunker.chunk_documents(self.documents, strategy=self.strategy)
        self.chunk_texts = chunker.chunks_to_texts(self.chunks)

        return self.chunks, self.chunk_texts

    def save_corpus(self, output_dir: str):
        """Save chunked corpus to multiple formats for fast loading."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        jsonl_path = output_path / "chunked_corpus.jsonl"
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for chunk in self.chunks:
                chunk_dict = {
                    "text": chunk.text,
                    "chunk_id": chunk.chunk_id,
                    "source_url": chunk.source_url,
                    "source_title": chunk.source_title,
                    "source_category": chunk.source_category,
                    "chunk_index": chunk.chunk_index,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "token_count": chunk.token_count
                }
                f.write(json.dumps(chunk_dict, ensure_ascii=False) + '\n')

        pkl_path = output_path / "chunked_corpus_texts.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump(self.chunk_texts, f)

        metadata = self.get_statistics()
        metadata_path = output_path / "corpus_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        return {
            "jsonl": str(jsonl_path),
            "pickle": str(pkl_path),
            "metadata": str(metadata_path)
        }

    def get_statistics(self) -> Dict:
        """Compute and return corpus statistics."""
        if not self.chunks:
            return {}

        token_counts = [chunk.token_count for chunk in self.chunks]
        char_counts = [len(chunk.text) for chunk in self.chunks]

        categories = {}
        for chunk in self.chunks:
            cat = chunk.source_category
            categories[cat] = categories.get(cat, 0) + 1

        return {
            "num_documents": len(self.documents),
            "num_chunks": len(self.chunks),
            "chunks_per_doc": len(self.chunks) / len(self.documents) if self.documents else 0,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "strategy": self.strategy,
            "token_stats": {
                "min": min(token_counts),
                "max": max(token_counts),
                "avg": sum(token_counts) / len(token_counts)
            },
            "char_stats": {
                "min": min(char_counts),
                "max": max(char_counts),
                "avg": sum(char_counts) / len(char_counts)
            },
            "categories": categories,
            "kb_path": str(self.kb_path)
        }


def build_and_save_corpus(kb_path: str, output_dir: str, chunk_size: int = 512,
                          overlap: int = 128, strategy: str = "rolling") -> Dict:
    """Build corpus from knowledge base and save to output directory."""
    builder = CorpusBuilder(kb_path, chunk_size, overlap, strategy)

    print(f"Loading knowledge base from: {kb_path}")
    docs = builder.load_knowledge_base()
    print(f"Loaded {len(docs)} documents")

    print(f"Chunking documents (size={chunk_size}, overlap={overlap}, strategy={strategy})...")
    chunks, texts = builder.build_corpus()
    print(f"Created {len(chunks)} chunks")

    print(f"Saving corpus to: {output_dir}")
    paths = builder.save_corpus(output_dir)

    stats = builder.get_statistics()
    print(f"Corpus statistics:")
    print(f"  Chunks per document: {stats['chunks_per_doc']:.2f}")
    print(f"  Avg tokens per chunk: {stats['token_stats']['avg']:.1f}")
    print(f"  Categories: {len(stats['categories'])}")

    return {
        "paths": paths,
        "statistics": stats
    }
