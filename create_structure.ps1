# PowerShell script to create RAG project structure

# Directories
$dirs = @(
    "rag_system",
    "rag_system\data",
    "rag_system\retrieval",
    "rag_system\reader",
    "rag_system\pipeline",
    "rag_system\utils"
)

foreach ($dir in $dirs) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

# Files
$files = @(
    "rag_system\__init__.py",
    "rag_system\data\__init__.py",
    "rag_system\data\chunker.py",
    "rag_system\data\corpus_builder.py",
    "rag_system\retrieval\__init__.py",
    "rag_system\retrieval\base_retriever.py",
    "rag_system\retrieval\bm25_retriever.py",
    "rag_system\retrieval\dense_retriever.py",
    "rag_system\retrieval\hybrid_retriever.py",
    "rag_system\reader\__init__.py",
    "rag_system\reader\base_reader.py",
    "rag_system\reader\qa_reader.py",
    "rag_system\reader\llm_reader.py",
    "rag_system\pipeline\__init__.py",
    "rag_system\pipeline\rag_pipeline.py",
    "rag_system\utils\__init__.py",
    "rag_system\utils\evaluation.py",
    "rag_system\utils\preprocessing.py",
    "rag_system\utils\logger.py"
)

foreach ($file in $files) {
    if (-not (Test-Path $file)) {
        New-Item -ItemType File -Force -Path $file | Out-Null
    }
}

Write-Host "Folder and files created successfully under ./rag_system/"
