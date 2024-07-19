from haystack.nodes.reader import BaseReader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from llama_index import StorageContext, load_index_from_storage

class FinetunedReader(BaseReader):
    def __init__(self):
        self.model = AutoModelForSeq2SeqLM.from_pretrained("./finetuned_model")
        self.tokenizer = AutoTokenizer.from_pretrained("./finetuned_model")

    def predict(self, query: str, documents, top_k: int):
        input_text = "question: " + query
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids
        outputs = self.model.generate(input_ids, max_length=100)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return [{"answer": answer, "score": 1.0}]

class LlamaIndexReader(BaseReader):
    def __init__(self, query_engine):
        self.query_engine = query_engine

    def predict(self, query: str, documents, top_k: int):
        response = self.query_engine.query(query)
        return [{"answer": str(response), "score": response.score}]