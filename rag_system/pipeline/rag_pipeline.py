class RAGPipeline:
    def __init__(self, retriever, reader, top_k=5):
        self.retriever = retriever
        self.reader = reader
        self.top_k = top_k

    def run(self, question: str):
        retrieved = self.retriever.retrieve(question, self.top_k)
        answer = self.reader.answer(question, retrieved)
        return {"question": question, "answer": answer, "contexts": retrieved}
