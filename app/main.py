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
    if not doc:
        return {"embedding": []}
    
    return {"embedding": doc.vector.tolist()}



# ===== CNN Inference Endpoint =====
from fastapi import UploadFile, File
from PIL import Image
import io, torch, torchvision.transforms as T
from models.cnn import SimpleCNN

# Device
device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")

# Load model weights (from artifacts/cnn_cifar10.pt)
_cnn = SimpleCNN(num_classes=10).to(device)
_cnn.load_state_dict(torch.load("artifacts/cnn_cifar10.pt", map_location=device))
_cnn.eval()

# Preprocess (CIFAR-10 style, 32x32)
_cnn_tfm = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor(),
    T.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010)),
])

CIFAR10_LABELS = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    x = _cnn_tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = _cnn(x)
        prob = torch.softmax(logits, dim=1)[0]
        idx = int(prob.argmax().item())
    return {
        "label": CIFAR10_LABELS[idx],
        "confidence": float(prob[idx].item())
    }
