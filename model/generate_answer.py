from transformers import pipeline
from embeddings.search import search

# Load QA pipeline
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

print("\n🧠 Ask a question (or type 'exit'):")

while True:
    question = input("🧠 Q: ")
    if question.lower() in {"exit", "quit"}:
        break

    print("🔍 Retrieving relevant docs...")
    results = search(question)

    if not results:
        print("❌ No relevant context found.")
        continue

    # Combine top results for context
    context = " ".join([r["abstract"] for r in results])

    # Run QA pipeline
    try:
        answer = qa_pipeline({
            "question": question,
            "context": context
        })
        print("💬 A:", answer["answer"])
    except Exception as e:
        print("❌ QA pipeline error:", str(e))
