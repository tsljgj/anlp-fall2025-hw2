"""
Document Chunker for RAG System
Implements multiple chunking strategies including rolling window chunking.
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Chunk:
    """Represents a chunk of text with metadata"""
    text: str
    chunk_id: str
    source_doc_id: str
    source_url: str
    source_title: str
    source_category: str
    chunk_index: int
    start_char: int
    end_char: int
    token_count: int


class DocumentChunker:
    """
    Chunks documents using various strategies.
    Primary method: Rolling window chunking (sliding window with overlap).
    """

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 128,
        tokenizer: str = "whitespace"
    ):
        """
        Initialize the chunker.

        Args:
            chunk_size: Maximum tokens per chunk
            overlap: Number of tokens to overlap between chunks
            tokenizer: Tokenization method ('whitespace' or 'simple')
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = tokenizer

        if overlap >= chunk_size:
            raise ValueError(f"Overlap ({overlap}) must be less than chunk_size ({chunk_size})")

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using specified tokenizer."""
        if self.tokenizer == "whitespace":
            return len(text.split())
        elif self.tokenizer == "simple":
            # Simple word boundary tokenization
            return len(re.findall(r'\b\w+\b', text))
        else:
            raise ValueError(f"Unknown tokenizer: {self.tokenizer}")

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex."""
        # Handle common abbreviations
        text = re.sub(r'\b(Dr|Mr|Mrs|Ms|Prof|Sr|Jr|vs|etc|Inc|Ltd|Corp)\.\s', r'\1<DOT> ', text)

        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Restore abbreviations
        sentences = [s.replace('<DOT>', '.') for s in sentences]

        return [s.strip() for s in sentences if s.strip()]

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Split on double newlines or markdown headers
        paragraphs = re.split(r'\n\s*\n|(?=^#{1,6}\s)', text, flags=re.MULTILINE)
        return [p.strip() for p in paragraphs if p.strip()]

    def rolling_window_chunking(
        self,
        text: str,
        preserve_paragraphs: bool = False
    ) -> List[Tuple[str, int, int]]:
        """
        Chunk text using rolling/sliding window approach.

        Args:
            text: Input text to chunk
            preserve_paragraphs: If True, try to respect paragraph boundaries

        Returns:
            List of tuples: (chunk_text, start_char, end_char)
        """
        if not text or not text.strip():
            return []

        # Tokenize the text
        tokens = text.split() if self.tokenizer == "whitespace" else re.findall(r'\b\w+\b', text)

        if len(tokens) <= self.chunk_size:
            # Document is smaller than chunk size, return as-is
            return [(text, 0, len(text))]

        chunks = []
        stride = self.chunk_size - self.overlap

        if stride <= 0:
            stride = self.chunk_size // 2  # Fallback to 50% overlap

        # Create overlapping windows
        for i in range(0, len(tokens), stride):
            chunk_tokens = tokens[i:i + self.chunk_size]

            if not chunk_tokens:
                break

            # Reconstruct text from tokens (approximate)
            # Find position in original text
            if self.tokenizer == "whitespace":
                chunk_text = " ".join(chunk_tokens)
                # Find approximate character positions
                search_text = " ".join(tokens[:i]) if i > 0 else ""
                start_char = len(search_text) + (1 if i > 0 else 0)
                end_char = start_char + len(chunk_text)
            else:
                chunk_text = " ".join(chunk_tokens)
                start_char = 0  # Simplified for now
                end_char = len(chunk_text)

            chunks.append((chunk_text, start_char, min(end_char, len(text))))

            # If we've covered the entire document, stop
            if i + self.chunk_size >= len(tokens):
                break

        return chunks

    def sentence_aware_chunking(
        self,
        text: str,
        min_sentences: int = 3
    ) -> List[Tuple[str, int, int]]:
        """
        Chunk text while respecting sentence boundaries.
        Groups sentences until token limit is reached.

        Args:
            text: Input text to chunk
            min_sentences: Minimum sentences per chunk (if possible)

        Returns:
            List of tuples: (chunk_text, start_char, end_char)
        """
        if not text or not text.strip():
            return []

        sentences = self._split_into_sentences(text)

        if not sentences:
            # Fallback to rolling window if no sentences found
            return self.rolling_window_chunking(text)

        chunks = []
        current_chunk = []
        current_tokens = 0
        start_pos = 0

        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)

            # If single sentence exceeds chunk size, split it
            if sentence_tokens > self.chunk_size:
                # Save current chunk if exists
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append((chunk_text, start_pos, start_pos + len(chunk_text)))
                    current_chunk = []
                    current_tokens = 0

                # Split long sentence using rolling window
                sentence_chunks = self.rolling_window_chunking(sentence)
                chunks.extend(sentence_chunks)
                start_pos += len(sentence)
                continue

            # Check if adding this sentence would exceed limit
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                chunks.append((chunk_text, start_pos, start_pos + len(chunk_text)))

                # Start new chunk with overlap (reuse last few sentences)
                overlap_sentences = []
                overlap_tokens = 0
                for s in reversed(current_chunk):
                    s_tokens = self._count_tokens(s)
                    if overlap_tokens + s_tokens <= self.overlap:
                        overlap_sentences.insert(0, s)
                        overlap_tokens += s_tokens
                    else:
                        break

                current_chunk = overlap_sentences + [sentence]
                current_tokens = overlap_tokens + sentence_tokens
                start_pos += len(chunk_text) - sum(len(s) for s in overlap_sentences)
            else:
                # Add sentence to current chunk
                current_chunk.append(sentence)
                current_tokens += sentence_tokens

        # Add remaining chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append((chunk_text, start_pos, start_pos + len(chunk_text)))

        return chunks

    def paragraph_based_chunking(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Chunk text by paragraphs, combining paragraphs until token limit.

        Args:
            text: Input text to chunk

        Returns:
            List of tuples: (chunk_text, start_char, end_char)
        """
        if not text or not text.strip():
            return []

        paragraphs = self._split_into_paragraphs(text)

        if not paragraphs:
            # Fallback to rolling window
            return self.rolling_window_chunking(text)

        chunks = []
        current_chunk = []
        current_tokens = 0
        start_pos = 0

        for para in paragraphs:
            para_tokens = self._count_tokens(para)

            # If single paragraph exceeds chunk size, use sentence-aware chunking
            if para_tokens > self.chunk_size:
                # Save current chunk if exists
                if current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    chunks.append((chunk_text, start_pos, start_pos + len(chunk_text)))
                    current_chunk = []
                    current_tokens = 0

                # Chunk the long paragraph
                para_chunks = self.sentence_aware_chunking(para)
                chunks.extend(para_chunks)
                start_pos += len(para)
                continue

            # Check if adding this paragraph would exceed limit
            if current_tokens + para_tokens > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = "\n\n".join(current_chunk)
                chunks.append((chunk_text, start_pos, start_pos + len(chunk_text)))

                # Start new chunk (paragraphs are discrete, less overlap needed)
                current_chunk = [para]
                current_tokens = para_tokens
                start_pos += len(chunk_text)
            else:
                # Add paragraph to current chunk
                current_chunk.append(para)
                current_tokens += para_tokens

        # Add remaining chunk
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append((chunk_text, start_pos, start_pos + len(chunk_text)))

        return chunks

    def semantic_chunking(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Semantic chunking: combines sentences with better context awareness.
        Similar to sentence_aware but with more intelligent grouping.

        Args:
            text: Input text to chunk

        Returns:
            List of tuples: (chunk_text, start_char, end_char)
        """
        # For now, use sentence-aware as the semantic strategy
        # In a more advanced implementation, this could use:
        # - Topic modeling
        # - Semantic similarity between sentences
        # - Named entity recognition for continuity
        return self.sentence_aware_chunking(text, min_sentences=2)

    def chunk_document(
        self,
        document: Dict,
        strategy: str = "rolling",
        doc_index: int = 0
    ) -> List[Chunk]:
        """
        Chunk a single document and return Chunk objects with metadata.

        Args:
            document: Document dict with 'content', 'url', 'title', etc.
            strategy: Chunking strategy ('rolling', 'sentence', 'paragraph', 'semantic')
            doc_index: Index of document in corpus

        Returns:
            List of Chunk objects
        """
        content = document.get("content", "")

        if not content or not content.strip():
            return []

        # Select chunking strategy
        if strategy == "rolling":
            raw_chunks = self.rolling_window_chunking(content)
        elif strategy == "sentence":
            raw_chunks = self.sentence_aware_chunking(content)
        elif strategy == "paragraph":
            raw_chunks = self.paragraph_based_chunking(content)
        elif strategy == "semantic":
            raw_chunks = self.semantic_chunking(content)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Convert to Chunk objects
        chunks = []
        for idx, (chunk_text, start_char, end_char) in enumerate(raw_chunks):
            chunk = Chunk(
                text=chunk_text,
                chunk_id=f"doc{doc_index}_chunk{idx}",
                source_doc_id=document.get("url", f"doc_{doc_index}"),
                source_url=document.get("url", ""),
                source_title=document.get("title", ""),
                source_category=document.get("source_category", ""),
                chunk_index=idx,
                start_char=start_char,
                end_char=end_char,
                token_count=self._count_tokens(chunk_text)
            )
            chunks.append(chunk)

        return chunks

    def chunk_documents(
        self,
        documents: List[Dict],
        strategy: str = "rolling"
    ) -> List[Chunk]:
        """
        Chunk multiple documents.

        Args:
            documents: List of document dicts
            strategy: Chunking strategy to use

        Returns:
            List of all Chunk objects from all documents
        """
        all_chunks = []

        for idx, doc in enumerate(documents):
            doc_chunks = self.chunk_document(doc, strategy=strategy, doc_index=idx)
            all_chunks.extend(doc_chunks)

        return all_chunks

    def chunks_to_texts(self, chunks: List[Chunk]) -> List[str]:
        """
        Convert Chunk objects to plain text list (for retriever compatibility).

        Args:
            chunks: List of Chunk objects

        Returns:
            List of chunk texts
        """
        return [chunk.text for chunk in chunks]

    def chunks_to_dicts(self, chunks: List[Chunk]) -> List[Dict]:
        """
        Convert Chunk objects to dict list with metadata.

        Args:
            chunks: List of Chunk objects

        Returns:
            List of chunk dicts
        """
        return [
            {
                "text": chunk.text,
                "chunk_id": chunk.chunk_id,
                "source_url": chunk.source_url,
                "source_title": chunk.source_title,
                "source_category": chunk.source_category,
                "chunk_index": chunk.chunk_index,
                "token_count": chunk.token_count
            }
            for chunk in chunks
        ]
