from transformers import AutoModel, AutoProcessor
from peft import LoraConfig, get_peft_model

def build_model_and_processor(model_id="Qwen/Qwen3.6"):
	processor = AutoProcessor.from_pretrained(model_id)

	# Load modelin bfloat16 to save memory
	model = AutoModel.from_pretrained(
		model_id,
		device_map="auto",
		torch_dtype="bfloat16"
	)

	# Freeze the vision tower explicity
	for param in model.visual_parameters():
		param.requires_grad = False
	
	# Apply LoRA to the LLM attention layers
	lora_config = LoraConfig(
		r=16,
		lora_alpha=32,
		target_modules=["q_proj", "k_proj", "o_proj", "v_proj"]
		lora_dropout=0.05,
		bias="none",
		tasl_type="CAUSAL_LM"
	)

	model = get_peft_model(model, lora_config)
	model.print_trainable_parameters()

	return model, processor

