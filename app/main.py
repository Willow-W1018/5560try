from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
import torch
from torchvision.utils import save_image
from models.gan import Generator
import os

app = FastAPI(title="GAN FastAPI Server")

device = "cuda" if torch.cuda.is_available() else "cpu"


G = Generator().to(device)
weights_path = "artifacts/generator_mnist.pt"

if os.path.exists(weights_path):
    G.load_state_dict(torch.load(weights_path, map_location=device))
    G.eval()
    print("Generator weights loaded successfully!")
else:
    print("Warning: generator_mnist.pt not found in artifacts/. Please train using train_gan.py first.")


@app.get("/", response_class=HTMLResponse)
def read_root():
    html_content = """
    <html>
        <head>
            <title>GAN API Running</title>
        </head>
        <body style="font-family: Arial; margin: 40px;">
            <h2>🚀 GAN API is running successfully!</h2>
            <p>Available endpoints:</p>
            <ul>
                <li><a href="/generate_digit_json" target="_blank">🧾 /generate_digit_json</a> — Generate a digit (JSON response)</li>
                <li><a href="/generate_digit_image" target="_blank">🖼️ /generate_digit_image</a> — Generate a digit (Image response)</li>
            </ul>
            <p>Swagger UI (API docs): <a href="/docs" target="_blank">/docs</a></p>
        </body>
    </html>
    """
    return html_content

# JSON
@app.get("/generate_digit_json")
def generate_digit_json():
    z = torch.randn(1, 100, device=device)
    img = G(z)
    os.makedirs("artifacts", exist_ok=True)
    out_path = "artifacts/generated_digit.png"
    save_image(img, out_path, normalize=True)
    print(f"Digit generated successfully at {out_path}")
    return {
        "message": "Digit generated successfully!",
        "noise_dim": 100,
        "output_path": out_path
    }

# Image
@app.get("/generate_digit_image")
def generate_digit_image():
    z = torch.randn(1, 100, device=device)
    img = G(z)
    os.makedirs("artifacts", exist_ok=True)
    out_path = "artifacts/generated_digit.png"
    save_image(img, out_path, normalize=True)
    print(f"Image available at {out_path}")
    return FileResponse(out_path, media_type="image/png")


# Reminder
print("FastAPI server is ready!")
print("   http://127.0.0.1:8000/")
print("   http://127.0.0.1:8000/generate_digit_json")
print("   http://127.0.0.1:8000/generate_digit_image")
print("API Docs: http://127.0.0.1:8000/docs")