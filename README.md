# 🌍 Environmental Health LLM QA System

A lightweight, end-to-end NLP application for question answering over a corpus of environmental health documents. Built using Hugging Face Transformers, SentenceTransformers, FAISS for retrieval, and the Hugging Face Inference API for LLM generation. Designed to demonstrate practical ML engineering and NLP deployment skills in a resource-efficient way.

---

## Project Summary

This project answers user questions about environmental health topics (e.g., wildfire smoke, air pollution, climate change) by:

1. Embedding a curated set of research abstracts and articles
2. Using FAISS to index and retrieve relevant documents
3. Passing relevant context + question to a hosted LLM for final answer generation

All components are lightweight and designed to run on a local machine (MacBook Air, no GPU) using a minimal, production-style pipeline.

---

## Architecture

**1. Data Processing**

* Curated \~100 research abstracts on environmental health
* Preprocessed into title + abstract JSON format

**2. Embedding + Indexing**

* Embeddings generated using: `sentence-transformers/all-MiniLM-L6-v2`
* Indexed with FAISS: `IndexFlatL2`

**3. Retrieval (RAG-style)**

* Top-K relevant documents retrieved via vector similarity (semantic search)
* Used as context for the generation step

**4. Question Answering**

* Contextual answers generated via Hugging Face Inference API
* Current model: `deepset/roberta-base-squad2` (question-answering)

**5. FastAPI Interface**

* Serve the QA pipeline via an HTTP API at `/ask`
* Swagger UI available at `/docs`

---

## Features

* **RAG-style Pipeline:** Retrieval-augmented generation architecture
* **LLM-as-a-Service:** Uses Hugging Face Inference API for LLMs (no need to fine-tune or host)
* **Scalable Indexing:** FAISS for fast nearest-neighbor search
* **FastAPI Web Interface** to expose the system over HTTP
* **Simple CLI Interface** for interactive use
* **Compatible with M1/M2 Macs and low-spec hardware**

---

## Use Case Examples

**Q:** What are the health risks of wildfire smoke?

**A:** Wildfire smoke is harmful to respiratory health. It can trigger asthma, bronchitis, and cardiovascular issues, especially in vulnerable populations.

**Q:** How does climate change affect disease spread?

**A:** Rising temperatures and changing rainfall patterns expand the habitats of disease-carrying insects like mosquitoes, increasing risk of diseases such as malaria or dengue.

---

## 📁 Repository Structure

```bash
.
├── data/                   # Source documents
├── embeddings/             # FAISS index + metadata
│   ├── build_index.py      # Create index
│   └── search.py           # Run semantic search
├── model/
│   └── generate_answer.py  # Full QA pipeline
├── app/
│   └── main.py             # FastAPI application
├── hf_test.py              # Test HF API integration
├── .env                    # Contains Hugging Face token (excluded from Git)
├── .gitignore              # Ensures .env and other files are excluded
└── requirements.txt
```

---

## How to Run

### CLI Mode

```bash
# Create conda env
conda create -n env_health_llm python=3.10
conda activate env_health_llm
pip install -r requirements.txt

# Add Hugging Face API token
echo "HF_API_TOKEN=your_token_here" > .env

# Step 1: Build embeddings
python embeddings/build_index.py

# Step 2: Ask questions via CLI
python -m model.generate_answer
```

### FastAPI Web Server

```bash
# Run the FastAPI app
uvicorn app.main:app --reload

# Access interactive docs
http://127.0.0.1:8000/docs
```

---

## Why This Project?

I built this project to:

* Solidify MLE skills across data pipelines, retrieval, and inference APIs
* Explore NLP for climate and public health
* Showcase end-to-end deployment skills, even on low-spec machines

---

## Future Improvements

* Swap out HF inference API for local model hosting
* Add Streamlit or web UI
* Expand corpus with PubMed + EPA data
* Improve re-ranking of retrieved documents
* Add logging, error handling, and batch processing

---

## Skills Demonstrated

* NLP: Embeddings, transformers, QA pipelines
* MLOps: Project structure, API usage, environment management
* Deployment: FastAPI server, CLI + web
* Domain interest: Climate + health

---

## Author

**Camilla Nawaz**
Machine Learning Engineer | Data Scientist | Climate + Health

[GitHub](https://github.com/camillan) • [LinkedIn](https://linkedin.com/in/camillanawaz)

---

*If you found this helpful or inspiring, feel free to star the repo or reach out!*
