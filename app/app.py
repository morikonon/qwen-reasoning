from pydantic import BaseModel, Upload
from fastapi import FastAPI
from model import inference

app = FastAPI("Qwen-math service")

@app.get("/")
async def work():
	return "Service is working!!!"

@app.post("/get_answer")
async def get_answer(UploadFile(...))