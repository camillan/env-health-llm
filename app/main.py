from fastapi import FastAPI
from pydantic import BaseModel
from model.generate_answer import generate_final_answer

app = FastAPI()

class Question(BaseModel):
    query: str  # ✅ Matches 'query' in the function

@app.get("/")
def read_root():
    return {"message": "Environmental Health LLM is running!"}

@app.post("/ask")
def ask_question(question: Question):
    try:
        answer = generate_final_answer(question.query)  # ✅ consistent
        print("Final Answer:", answer)
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}
