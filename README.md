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
uvicorn app.app:app --reload
streamlit run app/ui.py
'''

## Goal
The goal of this project is to explore how small language models can be fine-tuned to perform mathematical reasoning using explicit thinking-style responses.