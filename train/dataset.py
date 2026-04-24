import torch
from torch.utils.data import Dataset
from PIL import Image

class MathReasoningDataset(Dataset):
	def __init__(self, data_list, processor, max_length=1024):
		self.data_list = data_list
		self.processor = processor
		self.max_length = max_length
	
	def __len__(self):
		return len(self.data_list)

	def __getitem__(self, idx):
		item = self.data_list[idx]
		image = Image.open(item["image"]).convert("RGB")

		# Structure the prompt for Chain-of-Thought
		prompt = f"User: <image>\nSolve the math problem presented in the image. Think step-by-step.\Assistant: {item["long_answer"]}\nFinal Answer: {item["answer"]}"

		# The processor handles tokenizing text and extracting image patches
		inputs = self.processor(
			text=[prompt],
			images=[image],
			padding="max_length",
			max_length=self.max_length,
			return_tensors="pt"
		)

		# For causal LM, labels are usually the input_ids, padded_tokens set to -100
		labels = inputs["input_its"].clone()
		labels[labels == self.processor.tokenizer.pad_token_id] = -100
		inputs["labels"] = labels

		return {k: v.squeeze(0) for k, v in inputs.items()}