from transformers import pipeline
from embeddings.search import search
import requests
import os
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_API_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"

# Load QA pipeline
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

def generate_final_answer(query: str) -> str:  # <- renamed 'question' to 'query' for consistency
    print("🔍 Loading index and metadata...")
    docs = search(query)
    context = "\n".join([doc["abstract"] for doc in docs])

    if not context:
        return "❌ No relevant context found."

    payload = {"inputs": {"question": query, "context": context}}  # <- use 'query' here too
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    response = requests.post(API_URL, headers=headers, json=payload)

    print("Raw response:", response.json())  # Debug output

    if response.status_code == 200:
        try:
            return response.json()["answer"]
        except (KeyError, IndexError) as e:
            return f"❌ Could not extract answer: {str(e)}"
    else:
        return f"❌ LLM request failed: {response.status_code} - {response.text}"
    

# Optional CLI mode
if __name__ == "__main__":
    print("\n🧠 Ask a question (or type 'exit'):")
    while True:
        query = input("🧠 Q: ")  # <- renamed variable here too
        if query.lower() in {"exit", "quit"}:
            break
        answer = generate_final_answer(query)
        print("💬 A:", answer)
