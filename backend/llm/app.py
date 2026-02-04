from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama
import os

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "mistral-7b-instruct.gguf")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096
)

class ChatRequest(BaseModel):
    prompt: str

@app.post("/chat")
def chat(data: ChatRequest):
    response = llm(data.prompt, max_tokens=200)
    return {
        "answer": response["choices"][0]["text"]
    }
