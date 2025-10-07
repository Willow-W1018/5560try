from fastapi import FastAPI
from pydantic import BaseModel
import spacy

app = FastAPI()
nlp = spacy.load("en_core_web_sm")

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI with Docker & spaCy!"}

class GenReq(BaseModel):
    prompt: str

@app.post("/generate")
def generate_text(req: GenReq):
    return {"generated_text": f"Generated response to: {req.prompt}"}

class EmbReq(BaseModel):
    text: str

@app.post("/embed")
def get_embedding(req: EmbReq):
    doc = nlp(req.text)
    return {"embedding": doc.vector.tolist()}

