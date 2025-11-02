# Assignment 3

This project trains a Generative Adversarial Network (GAN) using PyTorch on the MNIST dataset and serves the trained generator through a FastAPI app running in Docker.

## Project Structure
app/main.py # FastAPI app

models/gan.py # Generator & Discriminator

train_gan.py # GAN training script

artifacts/ # Model weights & generated images

requirements.txt # Dependencies

Dockerfile # Docker setup

## Train the GAN
python train_gan.py

This saves the trained model to:
artifacts/generator_mnist.pt

## Run FastAPI Locally
uvicorn app.main:app --reload

Then open: http://127.0.0.1:8000

Endpoints:
/generate_digit_json → Returns JSON info

/generate_digit_image → Shows generated image

## Run with Docker
docker build -t gan-fastapi .

docker run -p 8000:8000 gan-fastapi

Then open in browser: http://127.0.0.1:8000

## Expected Files
artifacts/generator_mnist.pt – trained model

artifacts/generated_digit.png – sample output