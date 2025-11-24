from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from torchvision.utils import save_image
import torch, os, io
from PIL import Image
from app.models import Generator, LSTMTextGenerator, SimpleUNet
from fastapi.responses import JSONResponse

app = FastAPI(title="Assignment 4 API")

device = "cuda" if torch.cuda.is_available() else "cpu"

# ===== GAN =====
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


# ===== RNN =====
class TextReq(BaseModel):
    seed_text: str

@app.post("/generate_with_rnn")
def generate_with_rnn(req: TextReq):
    # mock demo output (replace with real model later)
    return {"generated_text": f"{req.seed_text} ...generated sequence"}

# ===== Diffusion =====
@app.get("/generate_diffusion")
def generate_diffusion():
    model = SimpleUNet(); x = torch.randn(1,1,28,28)
    out = model(x)
    save_image(out, "artifacts/generated_diffusion.png", normalize=True)
    return {"message": "Diffusion output saved", "path": "artifacts/generated_diffusion.png"}

@app.get("/", include_in_schema=False)
def read_root():
    return JSONResponse({
        "message": "FastAPI is running!",
        "next_step": "Please open /docs in your browser to test the API.",
        "example_url": "http://127.0.0.1:8000/docs"
    })