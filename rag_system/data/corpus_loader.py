import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional


class CorpusLoader:
    def __init__(self, data_dir: str = None):
        """Initialize corpus loader with data directory path."""
        if data_dir is None:
            data_dir = Path(__file__).parent / "processed"
        self.data_dir = Path(data_dir)
        self._cache = {}

    def load_texts(self, use_cache: bool = True) -> List[str]:
        """Load chunk texts only (fast) from pickle file."""
        if use_cache and "texts" in self._cache:
            return self._cache["texts"]

        pkl_path = self.data_dir / "chunked_corpus_texts.pkl"

        if not pkl_path.exists():
            raise FileNotFoundError(
                f"Chunked corpus not found at {pkl_path}. "
                f"Run scripts/build_corpus.py first."
            )

        with open(pkl_path, 'rb') as f:
            texts = pickle.load(f)

        if use_cache:
            self._cache["texts"] = texts

        return texts

    def load_full(self, use_cache: bool = True) -> List[Dict]:
        """Load full chunks with all metadata from JSONL file."""
        if use_cache and "full" in self._cache:
            return self._cache["full"]

        jsonl_path = self.data_dir / "chunked_corpus.jsonl"

        if not jsonl_path.exists():
            raise FileNotFoundError(
                f"Chunked corpus not found at {jsonl_path}. "
                f"Run scripts/build_corpus.py first."
            )

        chunks = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                chunk = json.loads(line)
                chunks.append(chunk)

        if use_cache:
            self._cache["full"] = chunks

        return chunks

    def load_metadata(self) -> Dict:
        """Load corpus metadata and statistics."""
        metadata_path = self.data_dir / "corpus_metadata.json"

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Corpus metadata not found at {metadata_path}. "
                f"Run scripts/build_corpus.py first."
            )

        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        return metadata

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict]:
        """Retrieve a specific chunk by its ID."""
        chunks = self.load_full()
        for chunk in chunks:
            if chunk.get("chunk_id") == chunk_id:
                return chunk
        return None

    def get_chunks_by_category(self, category: str) -> List[Dict]:
        """Get all chunks from a specific source category."""
        chunks = self.load_full()
        return [c for c in chunks if c.get("source_category") == category]

    def get_chunks_by_source(self, source_url: str) -> List[Dict]:
        """Get all chunks from a specific source URL."""
        chunks = self.load_full()
        return [c for c in chunks if c.get("source_url") == source_url]

    def clear_cache(self):
        """Clear the internal cache."""
        self._cache = {}

    def exists(self) -> bool:
        """Check if processed corpus exists."""
        pkl_path = self.data_dir / "chunked_corpus_texts.pkl"
        jsonl_path = self.data_dir / "chunked_corpus.jsonl"
        metadata_path = self.data_dir / "corpus_metadata.json"
        return pkl_path.exists() and jsonl_path.exists() and metadata_path.exists()
