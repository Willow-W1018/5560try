# Assignment 4 – GAN + RNN + Diffusion API

### Run locally
uvicorn app.main:app --reload

Open [http://127.0.0.1:8000/docs]
Note: The root URL (http://127.0.0.1:8000) will show "Not Found" — that’s normal!

### Endpoints
- `/generate_gan`
- `/generate_with_rnn`
- `/generate_diffusion`

### Docker
docker build -t sps-assignment4 .
docker run -p 8000:8000 sps-assignment4