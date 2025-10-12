Tangyi Qian (Jerry) — Chunking Framework and Comparative Evaluation
Overview

Tangyi Qian developed the document chunking pipeline, implemented an alternative RAG system, and built the performance benchmarking framework. His work provided the project’s data preprocessing and evaluation backbone.

Detailed Contributions

Document Chunker Development

Implemented chunker.py featuring four chunking strategies:

Rolling window – overlapping context-preserving chunks.

Sentence-aware – aligned to sentence boundaries for coherence.

Paragraph-based – coarse-grained segmentation for long texts.

Semantic – embedding-based segmentation by topic.

Built test_chunker.py to validate correctness, metadata preservation, and retrieval compatibility.

Supported configurable parameters for chunk size (default 512) and overlap (default 128).

Alternative RAG System Implementation

Built and evaluated an independent RAG system implementing sparse, dense, and hybrid retrievers.

Tuned hyperparameters and compared system-level performance with Wendy’s version.

Analyzed retrieval efficiency and accuracy trade-offs under different fusion weights.

End-to-End RAG Testing

Developed test_rag_system.py for full pipeline testing across all retrievers.

Processed 530 documents (1626 chunks) and tested on five benchmark CMU/Pittsburgh QA examples.

Logged context, timing, and accuracy to test_results.json, achieving 4 / 5 correct answers.

Co-authored with Claude Code for partial scaffolding and automated logging support.

Impact:
Tangyi’s work established a comprehensive evaluation and preprocessing foundation. His chunking framework and testing scripts ensured reproducibility, modularity, and quantitative comparison across RAG configurations.

Yiyang Wu (Wendy) — Repository Setup, RAG System Design, and Optimization
Overview

Wendy Wu established the initial project repository structure and led the design and implementation of the Retrieval-Augmented Generation (RAG) system, integrating sparse, dense, and hybrid retrievers. She introduced several optimization techniques, built testing pipelines, and created gold-standard QA data for evaluation.

Detailed Contributions

Repository and System Initialization

Established the initial project structure and organized module layout for retrieval, QA generation, and evaluation.

Set up the baseline retrieval and testing framework to support rapid iteration and team collaboration.

Configured environment files, directory hierarchy, and documentation for smooth development integration.

RAG System Implementation

Built a modular RAG system integrating:

Sparse retriever (BM25) for lexical matching.

Dense retriever (FAISS-based) for semantic similarity.

Hybrid retriever combining both modalities.

Implemented the QA reader that consumes retrieved contexts and produces grounded answers.

Designed interfaces for flexible comparison and evaluation between retriever types.

Hybrid Retriever Optimization

Added score normalization to ensure dense and sparse scores were comparable.

Integrated a reranker to refine the top-k retrieved passages and improve relevance precision.

Introduced dynamic alpha weighting, allowing adaptive balancing between lexical and semantic retrievers depending on query type and retriever confidence.

Improved retrieval precision, stability, and generalization across factual and descriptive QA tasks.

Testing and Evaluation

Developed comprehensive test scripts (test_rag.py) to benchmark sparse, dense, and hybrid retrieval modes.

Evaluated performance across latency, retrieval accuracy, and generated answer quality.

Created codes to build gold Question–Answer pairs from the web corpus for reliable downstream evaluation.

Documentation and Reporting

Authored the final technical report, detailing system architecture, retrieval algorithms, optimization strategies, and experimental findings.

Impact:
Wendy’s contributions laid both the architectural foundation and the core intelligence of the RAG system. Her work unified sparse and dense retrieval, introduced adaptive weighting and reranking, and established the testing and evaluation framework that guided the project’s development.


Zhihao Yuan — Data Collection, Integration, and Evaluation
Overview

Zhihao Yuan built the data collection and preprocessing pipeline that powers the RAG corpus and contributed significantly to dataset and system evaluation. His work ensured that all retrieval and QA models operated on a high-quality, diverse, and consistent knowledge base.

Detailed Contributions

Web Data Scraping and Preprocessing

Used Firecrawl to scrape and clean web data from multiple sources.

Processed all text into standardized JSONL format with fields:
['url', 'title', 'content', 'source_category', 'source_root_url', 'format', 'is_pdf', 'scraped_at', 'metadata'].

Filtered duplicates, removed noise, and normalized formatting for robust ingestion.

Integration with QA Pipeline

Adapted Wendy’s QA-generation code to integrate an LLM reader for automatic question–answer creation from the corpus.

Ensured contextual accuracy between retrieved passages and generated QA pairs.

Provided high-quality data that directly supported RAG evaluation.

Evaluation and Analysis

Performed dataset quality evaluation, assessing coverage, domain balance, and factual correctness.

Participated in RAG system evaluation, testing how preprocessing affected retriever recall and QA accuracy.

Helped analyze ablation study results, correlating data quality and retriever performance improvements.

Collaboration and Infrastructure Support

Coordinated merges and maintained consistency between branches.

Provided reproducible data documentation and interface utilities for downstream tasks.

Impact:
Zhihao’s pipeline and evaluation efforts formed the data and empirical foundation of the project. His preprocessing rigor and contributions to the ablation and performance studies ensured that results were both valid and interpretable.