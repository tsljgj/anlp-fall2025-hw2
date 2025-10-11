import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rag_system.data.corpus_builder import build_and_save_corpus


def main():
    """Build and save chunked corpus from knowledge base."""
    parser = argparse.ArgumentParser(
        description="Build chunked corpus from Pittsburgh/CMU knowledge base"
    )

    parser.add_argument(
        "--kb-path",
        type=str,
        default=None,
        help="Path to knowledge base JSONL file"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for processed corpus"
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Maximum tokens per chunk (default: 512)"
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=128,
        help="Number of tokens to overlap between chunks (default: 128)"
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default="rolling",
        choices=["rolling", "sentence", "paragraph", "semantic"],
        help="Chunking strategy (default: rolling)"
    )

    args = parser.parse_args()

    if args.kb_path is None:
        default_kb = Path(__file__).parent.parent.parent / "data_collection" / "data" / "pittsburgh_cmu_knowledge_base.jsonl"
        args.kb_path = str(default_kb)

    if args.output_dir is None:
        default_output = Path(__file__).parent.parent / "data" / "processed"
        args.output_dir = str(default_output)

    kb_path = Path(args.kb_path)
    if not kb_path.exists():
        print(f"Error: Knowledge base not found at {kb_path}")
        sys.exit(1)

    print("="*80)
    print("CORPUS BUILDER")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Knowledge base: {args.kb_path}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Chunk size: {args.chunk_size}")
    print(f"  Overlap: {args.overlap}")
    print(f"  Strategy: {args.strategy}")
    print()

    try:
        result = build_and_save_corpus(
            kb_path=args.kb_path,
            output_dir=args.output_dir,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            strategy=args.strategy
        )

        print("\n" + "="*80)
        print("SUCCESS!")
        print("="*80)
        print("\nSaved files:")
        for name, path in result["paths"].items():
            print(f"  {name}: {path}")

        print("\nYou can now use the corpus with:")
        print("  from rag_system.data.corpus_loader import CorpusLoader")
        print("  loader = CorpusLoader()")
        print("  corpus_texts = loader.load_texts()")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
