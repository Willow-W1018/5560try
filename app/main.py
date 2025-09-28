from fastapi import FastAPI
from pydantic import BaseModel
import spacy

app = FastAPI()

# 仅在启动时加载一次 spaCy 模型
nlp = spacy.load("en_core_web_sm")

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI with Docker & spaCy!"}

# 文本生成占位端点
class GenReq(BaseModel):
    prompt: str

@app.post("/generate")
def generate_text(req: GenReq):
    return {"generated_text": f"Generated response to: {req.prompt}"}

# 词/句嵌入端点
class EmbReq(BaseModel):
    text: str

@app.post("/embed")
def get_embedding(req: EmbReq):
    doc = nlp(req.text)
    if not doc:
        return {"embedding": []}
    # 句向量（Doc.vector）
    return {"embedding": doc.vector.tolist()}

