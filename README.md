## Repository Overview
The **main** branch contains code and materials from Assignment 1, Assignment 2 (early version), and several class activities. From Assignment 2, each assignment will have its own branch.


# Assignment 1 – FastAPI + Docker

uv run uvicorn app.main:app --reload

Then open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Run with Docker

docker build -t sps-genai .

docker run --rm -p 8000:8000 sps-genai

Then open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Quick test

curl http://127.0.0.1:8000/

curl -X POST "http://127.0.0.1:8000/embed" \
     -H "Content-Type: application/json" \
     -d '{"text": "This is a test."}'
