# Qwen Reasoning

This project fine-tunes the Qwen3.5-0.8B model on a mathematical reasoning dataset.

The main goal is to train the model to generate step-by-step reasoning before producing the final answer. This allows the model to better understand the task, decompose the problem, and solve it more accurately.

## Project Overview

The project includes:

- Fine-tuning Qwen3.5-0.8B on math reasoning data
- Training the model to produce reasoning-style answers
- FastAPI backend for model inference
- Streamlit UI for testing the model interactively

## How to Run This Project Locally

Clone the repository:

```bash
git clone https://github.com/morikonon/qwen-reasoning.git
cd qwen-reasoning

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run the API and UI in separate terminals (both from the repo root):

```bash
uvicorn app.app:app --reload
streamlit run app/ui.py
```

The Streamlit UI calls the FastAPI service over HTTP, so the API must be running first. By default the UI looks for the API at `http://localhost:8000`; override with the `API_URL` environment variable.

### Configuration

The API reads these environment variables:

- `BASE_MODEL` — base model repo id (default `Qwen/Qwen3.5-0.8B`)
- `ADAPTER_DIR` — path to the LoRA adapter weights (default `app/weights`)

The UI reads:

- `API_URL` — base URL of the FastAPI service (default `http://localhost:8000`)

## Running with Docker

```bash
docker compose up --build
```

This starts the API on `localhost:8000` and the Streamlit UI on `localhost:8051`. Note that 4-bit quantized inference requires a CUDA GPU; without one, the model loads in full bfloat16 on CPU, which is slow but functional.

## Training

```bash
python -m train.trainer
```

This fine-tunes the base model with LoRA on the `TIGER-Lab/VisualWebInstruct-Seed` dataset and writes checkpoints to `./checkpoint/qwen_math_reasoning`. Set `REPORT_TO=wandb` (and configure your W&B credentials) to enable experiment tracking; it defaults to `none`.

## Goal
The goal of this project is to explore how small language models can be fine-tuned to perform mathematical reasoning using explicit thinking-style responses.