from pydantic import BaseModel, UploadFile, File
from fastapi import FastAPI
from PIL import Image
from model import inference, load_model_and_processor
import io

app = FastAPI("Qwen-math service")

model, processor = load_model_and_processor("Qwen/Qwen3.5-0.8b", "weights/")

@app.get("/")
async def work():
	return "Service is working!!!"

@app.post("/get_answer")
async def caption_image(image_name: UploadFile = File(...)):
	contents = await image_name.read()

	try:
		image = Image.open(io.BytesIO(contents))
		caption = inference(model, processor, image, text_prompt="Solve this math task step by step and put the final answer into <answer> </answer> tags.")

		return f"generated_caption: {caption}"
		
	except Exception as e:
		return f"Error: {e}"