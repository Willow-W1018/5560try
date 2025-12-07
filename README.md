# Assignment 5 – QA API

This assignment adds a new **/qa** endpoint using a GPT-2 model fine-tuned on SQuAD.

## Run Locally

Before using `/qa`, please run:

python -m app.nlp.fine_tune_gpt2

This will create the fine-tuned model under models/gpt2-squad-formatted/, which is then used by the FastAPI app and Docker image.


Then:

uvicorn app.main:app --reload


Open:  
http://localhost:8000/docs

## Test /qa
POST:
json
{
  "question": "What is the capital of France?",
  "context": "France is a country in Europe. Paris is its capital."
}


Response format:


That is a great question. <answer> Let me know if you have any other questions.


## Docker


docker build -t assignment5 .
docker run -p 8000:8000 assignment5


## Files

* Fine-tuning script: `app/nlp/fine_tune_gpt2.py`
* Loader: `app/nlp/qa_model_loader.py`
* Model: `models/gpt2-squad-formatted/`
* Endpoint: `app/main.py`