import io
import os
import re
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image
from pydantic import BaseModel

from .model import load_model_and_processor, run_inference

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen3.5-0.8B")
ADAPTER_DIR = os.getenv("ADAPTER_DIR", os.path.join(BASE_DIR, "weights"))
DEFAULT_PROMPT = (
    "Solve this math problem step by step. Wrap your reasoning in <think></think> "
    "tags and put the final exact answer in <answer></answer> tags."
)

app = FastAPI(title="Qwen Math Reasoning Service")

model, processor = load_model_and_processor(BASE_MODEL, ADAPTER_DIR)


class AnswerResponse(BaseModel):
    think: str
    answer: str
    raw: str


def parse_response(text: str) -> AnswerResponse:
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)

    return AnswerResponse(
        think=think_match.group(1).strip() if think_match else "",
        answer=answer_match.group(1).strip() if answer_match else "",
        raw=text,
    )


@app.get("/")
async def health():
    return {"status": "ok"}


@app.post("/get_answer", response_model=AnswerResponse)
async def get_answer(image: UploadFile = File(...), prompt: Optional[str] = Form(DEFAULT_PROMPT)):
    contents = await image.read()

    try:
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        raw_output = run_inference(model, processor, pil_image, prompt or DEFAULT_PROMPT)
    except Exception as exc:
        return AnswerResponse(think="", answer="", raw=f"Error: {exc}")

    return parse_response(raw_output)
