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
                "content": "You are a helpful assistant that answers questions based only on the provided context. Be concise and factual. If the answer is not in the context, say so."
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