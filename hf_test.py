import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_TOKEN = os.getenv("HF_API_TOKEN")

API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-small"
print(API_URL)
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query_llm(question, context):
    prompt = f"""Answer the question using only the context below.

Context:
{context}

Q: {question}
A:"""

    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
    print("Status:", response.status_code)
    print("Result:", response.json())

# Try it
question = "What are the health effects of wildfire smoke?"
context = "Wildfire smoke is harmful to respiratory health and can trigger asthma, bronchitis, and other conditions."

query_llm(question, context)
