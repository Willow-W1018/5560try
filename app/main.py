from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from torchvision.utils import save_image
from PIL import Image
import torch, os, io
from app.models import Generator, LSTMTextGenerator, SimpleUNet
from app.nlp.qa_model_loader import load_qa_pipeline, ANSWER_PREFIX, ANSWER_SUFFIX


# FASTAPI INITIALIZATION
app = FastAPI(title="Assignment 5 API")

device = "cuda" if torch.cuda.is_available() else "cpu"


# Load Fine-Tuned GPT-2 for QA
try:
    qa_generator = load_qa_pipeline()
except Exception as e:
    print("❌ Failed to load GPT-2 QA model:", e)
    qa_generator = None


# Assignment 5: QA API
class QARequest(BaseModel):
    question: str
    context: str


@app.post("/qa")
def qa_endpoint(req: QARequest):
    """
    Answer a question using the fine-tuned GPT-2 model on SQuAD.
    """

    if qa_generator is None:
        return {"error": "GPT-2 QA model not loaded. Please run fine_tune_gpt2.py first. Check model path / Docker build."}

    prompt = (
        f"Question: {req.question}\n"
        f"Context: {req.context}\n"
        f"Answer: {ANSWER_PREFIX}"
    )

    outputs = qa_generator(
        prompt,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        num_return_sequences=1,
    )

    full_text = outputs[0]["generated_text"]
    answer = full_text[len(prompt):]

    if not answer.strip().endswith(ANSWER_SUFFIX.strip()):
        answer = answer.strip() + ANSWER_SUFFIX

    return {"answer": answer}


# GAN GENERATION
@app.get("/generate_gan")
def generate_gan():
    z = torch.randn(1, 100, device=device)
    G = Generator()

    model_path = "artifacts/generator_mnist.pt"
    if not os.path.exists(model_path):
        return {"error": f"Model file not found at {model_path}"}

    state_dict = torch.load(model_path, map_location=device)
    G.load_state_dict(state_dict, strict=False)
    G.eval()

    img = G(z)
    os.makedirs("artifacts", exist_ok=True)
    save_image(img, "artifacts/generated_digit.png", normalize=True)

    return {"message": "GAN image generated", "path": "artifacts/generated_digit.png"}


# RNN GENERATION
class TextReq(BaseModel):
    seed_text: str

@app.post("/generate_with_rnn")
def generate_with_rnn(req: TextReq):
    return {"generated_text": f"{req.seed_text} ...generated sequence"}


# DIFFUSION GENERATION
@app.get("/generate_diffusion")
def generate_diffusion():
    model = SimpleUNet()
    x = torch.randn(1, 1, 28, 28)
    out = model(x)

    save_image(out, "artifacts/generated_diffusion.png", normalize=True)

    return {"message": "Diffusion output saved", "path": "artifacts/generated_diffusion.png"}


# ROOT
@app.get("/", include_in_schema=False)
def read_root():
    return JSONResponse({
        "message": "FastAPI is running!",
        "next_step": "Open /docs in your browser to test the API.",
        "example_url": "http://127.0.0.1:8000/docs"
    })