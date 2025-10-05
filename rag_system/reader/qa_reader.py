from transformers import pipeline
from typing import List, Dict

class QAReader:
    def __init__(self, model_name="distilbert-base-cased-distilled-squad"):
        self.qa_pipeline = pipeline("question-answering", model=model_name)

    def answer(self, question: str, contexts: List[Dict]) -> str:
        answers = []
        for ctx in contexts:
            result = self.qa_pipeline(question=question, context=ctx['text'])
            answers.append((result['answer'], result['score']))
        return max(answers, key=lambda x: x[1])[0]
