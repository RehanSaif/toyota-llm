# Toyota Manual QA System

A question-answering system for Toyota manuals using document retrieval and language model techniques.

## Features
- Document processing/indexing (Haystack, LlamaIndex)
- T5 model finetuning on Toyota manual data
- Dual QA approach (finetuned model + LlamaIndex)
- FastAPI web service

## Prerequisites
- Python 3.8+
- pip

## Setup & Usage

1. Clone and install:
git clone https://github.com/RehanSaif/toyota-llm.git
cd app
pip install -r requirements.txt

2. Place Toyota manuals in `data/` directory.

3. Run the app:
   python main.py
4. Query the system:
curl -X POST "http://localhost:8000/answer" -H "Content-Type: application/json" -d '{"question":"How often should I change the oil in my Toyota?"}'

## File Structure
- `data_processing.py`: Document processing and indexing
- `model_finetuning.py`: Model finetuning
- `custom_readers.py`: Custom Haystack readers
- `main.py`: FastAPI app setup

Note: First run may be slow due to document processing, indexing, and model finetuning.
