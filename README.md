# 🌍 Environmental Health LLM QA System

A lightweight, end-to-end NLP application for question answering over a corpus of environmental health documents. Built using Hugging Face Transformers, SentenceTransformers, FAISS for retrieval, and the Hugging Face Inference API for LLM generation. Designed to demonstrate practical ML engineering and NLP deployment skills in a resource-efficient way.

---

## 🧬 Project Summary

This project answers user questions about environmental health topics (e.g., wildfire smoke, air pollution, climate change) by:

1. Embedding a curated set of research abstracts and articles
2. Using FAISS to index and retrieve relevant documents
3. Passing relevant context + question to a hosted LLM for final answer generation

All components are lightweight and designed to run on a local machine (MacBook Air, no GPU) using a minimal, production-style pipeline.

---

## 🛠️ Architecture

**1. Data Processing**

* Curated \~100 research abstracts on environmental health
* Preprocessed into title + abstract JSON format

**2. Embedding + Indexing**

* Embeddings generated using: `sentence-transformers/all-MiniLM-L6-v2`
* Indexed with FAISS: `IndexFlatL2`

**3. Retrieval**

* Top-K relevant documents retrieved via cosine similarity

**4. Question Answering**

* Contextual answers generated via Hugging Face Inference API
* Current model: `facebook/bart-large-cnn` (text2text-generation)

---

## Features

* **RAG-style Pipeline:** Retrieval-augmented generation architecture
* **LLM-as-a-Service:** Uses Hugging Face Inference API for LLMs (no need to fine-tune or host)
* **Scalable Indexing:** FAISS for fast nearest-neighbor search
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
├── data/                 # Source documents
├── embeddings/           # FAISS index + metadata
│   ├── build_index.py    # Create index
│   └── search.py         # Run semantic search
├── model/
│   └── generate_answer.py  # Full QA pipeline
├── hf_test.py            # Test HF API integration
├── .env                  # Contains Hugging Face token (excluded from Git)
└── requirements.txt
```

---

## 🚀 How to Run

```bash
# Create conda env
conda create -n env_health_llm python=3.10
conda activate env_health_llm
pip install -r requirements.txt

# Add Hugging Face API token
echo "HF_API_TOKEN=your_token_here" > .env

# Step 1: Build embeddings
python embeddings/build_index.py

# Step 2: Ask questions
python -m model.generate_answer
```

---

## Why This Project?

I built this project to:

* Solidify MLE skills across data pipelines, retrieval, and inference APIs
* Explore NLP for climate and public health
* Showcase end-to-end deployment skills, even on low-spec machines

---

## Future Improvements

* Swap out HF inference API for local model hosting with FastAPI
* Add Streamlit or web UI
* Expand corpus with PubMed + EPA data
* Improve re-ranking of retrieved documents

---

## Skills Demonstrated

* NLP: Embeddings, transformers, summarization
* MLOps: Project structure, API usage, environment management
* Deployment: Token-safe repo, CLI interface, minimal memory footprint
* Domain interest: Climate + health

---

## Author

**Camilla Nawaz**
Machine Learning Engineer | Data Scientist | Machine Learning & AI Engineering

[GitHub](https://github.com/camillan) • [LinkedIn](https://linkedin.com/in/camillanawaz)

---

*If you found this helpful or inspiring, feel free to star the repo or reach out!*
