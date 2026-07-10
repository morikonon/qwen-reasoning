from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import LoraConfig, get_peft_model

DEFAULT_MODEL_ID = "Qwen/Qwen3.5-0.8B"


def build_model_and_processor(model_id=DEFAULT_MODEL_ID):
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    # Load model in bfloat16 to save memory
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="bfloat16",
        trust_remote_code=True
    )

    # Freeze the vision tower explicitly
    for name, param in model.named_parameters():
        if "visual" in name:
            param.requires_grad = False

    # Apply LoRA to the LLM attention layers
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "o_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, processor
