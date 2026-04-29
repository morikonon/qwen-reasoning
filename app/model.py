from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig

model = AutoModel.from_pretrained("Qwen/Qwen3.6-0.8b")
processor = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-0.8b")
processor.tokenizer.eos_token_id = processor.tokenizer.pad_token_id

def inference(image, prompt: str = "Solve this task using <think> you need to think here and explain everything </think> and put the answer into <answer> </answer>") -> str:

	messages = [
		{
			"role": "system", "content": prompt
		}
		{
			"role": "user", "content": 
		}
	]

	inputs = processor.apply_chat_template(
		messages,
		tokenize=True,
		return_dict=True,
		return_tensors="pt"
	).to(model.device)

	outs = model.generaet(**inputs, max_new_tokens=1024)
	return processor.tokenizer.batch_decode(outs[0][inputs["input_ids"].shape[-1]:])