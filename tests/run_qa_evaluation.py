import json
import re
import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_system.data.corpus_loader import CorpusLoader
from rag_system.retrieval.bm25_retriever import BM25Retriever
from rag_system.retrieval.dense_retriever import DenseRetriever
from rag_system.retrieval.hybrid_retriever import HybridRetriever
from rag_system.reader.llm_reader import LLMReader
from rag_system.pipeline.rag_pipeline import RAGPipeline


def load_questions(filepath: str) -> List[str]:
    """Load questions from text file (one per line)."""
    with open(filepath, 'r', encoding='utf-8') as f:
        questions = [line.strip() for line in f if line.strip()]
    return questions


def load_reference_answers(filepath: str) -> Dict[str, List[str]]:
    """Load reference answers from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_answers = json.load(f)

    reference_answers = {}
    for qid, answer_str in raw_answers.items():
        answers = [a.strip() for a in answer_str.split(';')]
        reference_answers[qid] = answers

    return reference_answers


def normalize_answer(text: str) -> str:
    """Normalize answer text for evaluation (SQuAD-style)."""
    text = text.lower()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = ' '.join(text.split())
    return text


def get_tokens(text: str) -> List[str]:
    """Get normalized tokens from text."""
    return normalize_answer(text).split()


def compute_exact_match(prediction: str, references: List[str]) -> int:
    """Compute exact match score (1 if any reference matches, 0 otherwise)."""
    normalized_pred = normalize_answer(prediction)
    for ref in references:
        normalized_ref = normalize_answer(ref)
        if normalized_pred == normalized_ref:
            return 1
    return 0


def compute_f1(prediction: str, references: List[str]) -> float:
    """Compute F1 score (max F1 across all references)."""
    pred_tokens = get_tokens(prediction)
    if not pred_tokens:
        return 0.0

    max_f1 = 0.0
    for reference in references:
        ref_tokens = get_tokens(reference)
        if not ref_tokens:
            continue

        common = Counter(pred_tokens) & Counter(ref_tokens)
        num_common = sum(common.values())

        if num_common == 0:
            f1 = 0.0
        else:
            precision = num_common / len(pred_tokens)
            recall = num_common / len(ref_tokens)
            f1 = 2 * precision * recall / (precision + recall)

        max_f1 = max(max_f1, f1)

    return max_f1


def compute_answer_recall(prediction: str, references: List[str]) -> int:
    """Compute answer recall (1 if any reference token in prediction, 0 otherwise)."""
    pred_tokens = set(get_tokens(prediction))
    if not pred_tokens:
        return 0

    for reference in references:
        ref_tokens = set(get_tokens(reference))
        if ref_tokens and pred_tokens & ref_tokens:
            return 1

    return 0


def setup_rag_pipeline(retriever_type: str, top_k: int, api_key: str = None) -> RAGPipeline:
    """Initialize and return RAG pipeline with specified retriever."""
    print(f"\n{'='*80}")
    print("INITIALIZING RAG PIPELINE")
    print(f"{'='*80}")

    print("\nLoading pre-chunked corpus...")
    loader = CorpusLoader()

    if not loader.exists():
        print("Error: Processed corpus not found.")
        print("Please run: python rag_system/scripts/build_corpus.py")
        sys.exit(1)

    corpus_texts = loader.load_texts()
    metadata = loader.load_metadata()
    print(f"Loaded {len(corpus_texts)} chunks from {metadata['num_documents']} documents")

    print(f"\nBuilding {retriever_type} retriever...")

    if retriever_type == "bm25":
        retriever = BM25Retriever(corpus_texts, name="BM25")
    elif retriever_type == "dense":
        retriever = DenseRetriever(corpus_texts, model_name="all-MiniLM-L6-v2", name="Dense")
    elif retriever_type == "hybrid":
        bm25 = BM25Retriever(corpus_texts, name="BM25")
        dense = DenseRetriever(corpus_texts, model_name="all-MiniLM-L6-v2", name="Dense")
        retriever = HybridRetriever(dense, bm25, alpha=0.6, name="Hybrid")
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")

    print("Retriever built successfully")

    print("\nSetting up LLM reader...")

    if not api_key:
        try:
            test_rag_path = Path(__file__).parent.parent / "test_rag.py"
            with open(test_rag_path, 'r') as f:
                content = f.read()
                match = re.search(r'TOGETHER_API_KEY\s*=\s*["\']([^"\']+)["\']', content)
                if match:
                    api_key = match.group(1)
        except:
            pass

    if not api_key:
        print("Warning: No API key found. Please set TOGETHER_API_KEY.")
        sys.exit(1)

    reader = LLMReader(model_name="Qwen/Qwen2.5-Coder-32B-Instruct", api_key=api_key)
    print("LLM reader ready")

    pipeline = RAGPipeline(retriever, reader, top_k=top_k)

    print(f"\n{'='*80}")
    print("RAG PIPELINE READY")
    print(f"{'='*80}\n")

    return pipeline


def run_qa_pipeline(questions: List[str], pipeline: RAGPipeline) -> Tuple[Dict[str, str], List[Dict]]:
    """Run RAG pipeline on all questions."""
    print(f"Processing {len(questions)} questions...\n")

    system_outputs = {}
    detailed_results = []

    for i, question in enumerate(questions, 1):
        qid = str(i)
        print(f"[{i}/{len(questions)}] {question}")

        try:
            start_time = time.time()
            result = pipeline.run(question)
            elapsed = time.time() - start_time

            answer = result['answer']
            system_outputs[qid] = answer

            detailed_results.append({
                "question_id": qid,
                "question": question,
                "answer": answer,
                "contexts": result['contexts'][:3],
                "time": elapsed
            })

            print(f"  Answer: {answer[:100]}..." if len(answer) > 100 else f"  Answer: {answer}")
            print(f"  Time: {elapsed:.2f}s")

        except Exception as e:
            print(f"  Error: {e}")
            system_outputs[qid] = ""
            detailed_results.append({
                "question_id": qid,
                "question": question,
                "answer": "",
                "error": str(e),
                "time": 0
            })

        print()

    return system_outputs, detailed_results


def evaluate_results(system_outputs: Dict[str, str], reference_answers: Dict[str, List[str]]) -> Dict:
    """Evaluate system outputs against reference answers."""
    print(f"\n{'='*80}")
    print("EVALUATION")
    print(f"{'='*80}\n")

    em_scores = []
    f1_scores = []
    recall_scores = []
    per_question_results = []

    for qid in sorted(system_outputs.keys(), key=int):
        prediction = system_outputs[qid]
        references = reference_answers.get(qid, [""])

        em = compute_exact_match(prediction, references)
        f1 = compute_f1(prediction, references)
        recall = compute_answer_recall(prediction, references)

        em_scores.append(em)
        f1_scores.append(f1)
        recall_scores.append(recall)

        per_question_results.append({
            "question_id": qid,
            "prediction": prediction,
            "references": references,
            "exact_match": em,
            "f1": f1,
            "answer_recall": recall
        })

    overall_metrics = {
        "exact_match": sum(em_scores) / len(em_scores) * 100 if em_scores else 0,
        "f1": sum(f1_scores) / len(f1_scores) * 100 if f1_scores else 0,
        "answer_recall": sum(recall_scores) / len(recall_scores) * 100 if recall_scores else 0,
        "total_questions": len(system_outputs)
    }

    return {
        "overall": overall_metrics,
        "per_question": per_question_results
    }


def print_evaluation_report(eval_results: Dict):
    """Print formatted evaluation report."""
    overall = eval_results["overall"]

    print(f"\n{'='*80}")
    print("EVALUATION RESULTS")
    print(f"{'='*80}\n")

    print(f"Total Questions: {overall['total_questions']}")
    print(f"\nOverall Metrics:")
    print(f"  Exact Match (EM):    {overall['exact_match']:.2f}%")
    print(f"  F1 Score:            {overall['f1']:.2f}%")
    print(f"  Answer Recall:       {overall['answer_recall']:.2f}%")

    print(f"\n{'='*80}")

    per_q = eval_results["per_question"]
    correct_em = sum(1 for q in per_q if q["exact_match"] == 1)
    print(f"\nExact Matches: {correct_em}/{len(per_q)} questions")

    f1_over_50 = sum(1 for q in per_q if q["f1"] >= 0.5)
    print(f"F1 >= 50%: {f1_over_50}/{len(per_q)} questions")

    print(f"\n{'='*80}\n")


def save_results(system_outputs: Dict, eval_results: Dict, detailed_results: List[Dict],
                 output_file: str, eval_file: str, detailed_file: str):
    """Save all results to files."""
    print("Saving results...")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(system_outputs, f, indent=2, ensure_ascii=False)
    print(f"  System outputs saved to: {output_file}")

    with open(eval_file, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    print(f"  Evaluation results saved to: {eval_file}")

    with open(detailed_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    print(f"  Detailed results saved to: {detailed_file}")

    print("\nAll results saved successfully!")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Run QA evaluation on test set"
    )

    parser.add_argument(
        "--retriever",
        type=str,
        default="hybrid",
        choices=["bm25", "dense", "hybrid"],
        help="Retriever type to use (default: hybrid)"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of documents to retrieve (default: 5)"
    )

    parser.add_argument(
        "--output-file",
        type=str,
        default="system_output_test.json",
        help="Output file for system answers (default: system_output_test.json)"
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Together API key (optional, will try to read from test_rag.py)"
    )

    args = parser.parse_args()

    tests_dir = Path(__file__).parent

    questions_file = tests_dir / "questions.txt"
    reference_file = tests_dir / "reference_answers.json"
    output_file = tests_dir / args.output_file
    eval_file = tests_dir / "evaluation_results.json"
    detailed_file = tests_dir / "detailed_results.json"

    print(f"\n{'='*80}")
    print("QA EVALUATION SCRIPT")
    print(f"{'='*80}\n")

    print(f"Configuration:")
    print(f"  Retriever: {args.retriever}")
    print(f"  Top-K: {args.top_k}")
    print(f"  Questions: {questions_file}")
    print(f"  References: {reference_file}")
    print(f"  Output: {output_file}")

    if not questions_file.exists():
        print(f"\nError: Questions file not found: {questions_file}")
        sys.exit(1)

    if not reference_file.exists():
        print(f"\nError: Reference answers file not found: {reference_file}")
        sys.exit(1)

    questions = load_questions(str(questions_file))
    reference_answers = load_reference_answers(str(reference_file))

    print(f"\nLoaded {len(questions)} questions")
    print(f"Loaded {len(reference_answers)} reference answers")

    pipeline = setup_rag_pipeline(args.retriever, args.top_k, args.api_key)

    system_outputs, detailed_results = run_qa_pipeline(questions, pipeline)

    eval_results = evaluate_results(system_outputs, reference_answers)

    print_evaluation_report(eval_results)

    save_results(
        system_outputs,
        eval_results,
        detailed_results,
        str(output_file),
        str(eval_file),
        str(detailed_file)
    )

    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
