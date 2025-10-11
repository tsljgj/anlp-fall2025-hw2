import os
from together import Together
from typing import List, Dict

class LLMReader:
    def __init__(self, model_name="Qwen/Qwen2.5-Coder-32B-Instruct", api_key=None):
        self.model_name = model_name
        self.client = Together(api_key=api_key or os.environ.get("TOGETHER_API_KEY"))

    def answer(self, question: str, contexts: List[Dict], max_tokens: int = 150) -> str:
        context_text = "\n\n".join([f"Document {i+1}: {ctx['text']}" for i, ctx in enumerate(contexts)])
        
        messages = [
            {
                "role": "system",
                "content": """You are a helpful assistant that answers questions based only on the provided context.

IMPORTANT: Provide SHORT, DIRECT answers only. Extract just the key fact, name, number, or phrase that answers the question.

Rules:
- NO preambles like "According to the context..." or "The answer is..."
- NO unnecessary elaboration or full sentences unless the question asks for explanation
- For who/what/when/where questions: provide just the name, thing, date, or place
- For yes/no questions: answer "Yes" or "No" with minimal justification if needed
- For list questions: provide a semicolon-separated list (e.g., "Item1; Item2; Item3")
- If the answer is not in the context, say "The answer is not available in the given context."

Examples:
- Q: "Who won the prize?" → A: "John Smith" (NOT "John Smith won the prize")
- Q: "How many people attended?" → A: "500" (NOT "500 people attended")
- Q: "When did it happen?" → A: "2025" or "October 15, 2025" (NOT "It happened in 2025")
- Q: "What are the three items?" → A: "Apple; Banana; Orange" (NOT "The three items are apple, banana, and orange")"""
            },
            {
                "role": "user",
                "content": f"""Context:
{context_text}

Question: {question}

Answer:"""
            }
        ]
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.1,
            top_p=0.9
        )
        
        answer = response.choices[0].message.content.strip()
        return answer