from fastapi import FastAPI
from pydantic import BaseModel
from haystack import Pipeline
from haystack.document_stores import WeaviateDocumentStore
from haystack.nodes import DensePassageRetriever
from custom_readers import FinetunedReader, LlamaIndexReader
from llama_index import StorageContext, load_index_from_storage
from data_processing import process_documents, prepare_finetuning_data
from model_finetuning import finetune_model
import uvicorn

app = FastAPI()

# Process documents and prepare data
manual_path = "path/to/toyota/manuals"
document_store, retriever = process_documents(manual_path)
qa_pairs = prepare_finetuning_data(document_store)

# Finetune model
finetune_model(qa_pairs)

# Initialize components
finetuned_reader = FinetunedReader()

# Load LlamaIndex
storage_context = StorageContext.from_defaults(persist_dir="toyota_index")
loaded_index = load_index_from_storage(storage_context)
query_engine = loaded_index.as_query_engine()
llama_reader = LlamaIndexReader(query_engine)

# Create pipeline
pipe = Pipeline()
pipe.add_node(component=retriever, name="Retriever", inputs=["Query"])
pipe.add_node(component=finetuned_reader, name="FinetunedReader", inputs=["Retriever"])
pipe.add_node(component=llama_reader, name="LlamaReader", inputs=["Retriever"])

class Query(BaseModel):
    question: str

@app.post("/answer")
def answer_question(query: Query):
    results = pipe.run(query=query.question)
    finetuned_answer = results["FinetunedReader"][0].answer
    llama_answer = results["LlamaReader"][0].answer
    return {
        "finetuned_answer": finetuned_answer,
        "llama_answer": llama_answer
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)