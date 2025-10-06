# PowerShell script to set up a RAG environment using uv (CPU-only, Windows-safe)

Write-Host "Setting up RAG environment with uv..."

# 1. Check if uv is installed
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv..."
    pip install uv
}

# 2. Create the environment (named 'rag')
Write-Host "Creating virtual environment 'rag'..."
uv venv rag --python 3.10

# 3. Activate it for this session
Write-Host "Activating environment..."
& .\rag\Scripts\activate.ps1

# 4. Upgrade pip and base build tools inside uv
Write-Host "Upgrading pip, setuptools, and wheel..."
uv pip install -U pip setuptools wheel

# 5. Install core scientific libraries
Write-Host "Installing core utilities..."
uv pip install numpy scipy pandas

# 6. Install NLP and retrieval dependencies (CPU-only)
Write-Host "Installing torch, transformers, and retrieval packages..."
uv pip install torch transformers sentence-transformers rank-bm25 faiss-cpu datasets tqdm colorama

# 7. Install spaCy (stable prebuilt version to avoid compiler issues)
Write-Host "Installing spaCy (stable Windows wheel)..."
pip install spacy==3.7.4

# 8. Download spaCy English model
Write-Host "Downloading spaCy English model..."
python -m spacy download en_core_web_sm

# 9. Verify installation
Write-Host "Verifying installation..."
python -c "import torch, spacy, faiss, transformers, sentence_transformers, numpy, pandas, tqdm; print('All libraries imported successfully. Torch using CUDA:', torch.cuda.is_available())"

Write-Host "Environment 'rag' setup completed."
Write-Host "To activate later, run: .\rag\Scripts\activate.ps1"
