import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

def load_model_and_processor(base_model_path, adapter_path):

    bits_and_bytes_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_double_compute=True,
        bnb_4bit_compute_type=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )
    # For Qwen VL models, use AutoModelForVision2Seq
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    
    model = AutoModelForImageTextToText.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        quantization_config=bits_and_bytes_config
    )
    
    # Load the LoRA adapters
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    
    return model, processor

def run_inference(model, processor, image, text_prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant with thinking capabilities."},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": text_prompt}
        ]}
    ]
    
    # Standard Qwen-VL template application
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=1024)
        
    # Trim the input tokens from the output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    return processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]