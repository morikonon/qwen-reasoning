from transformers import Trainer, TrainingArguments
from .dataset import MathReasoningDataset
from .model_builder import build_model_and_processor
from .metrics import build_compute_metrics_fn

def run_training():
	model, processor = build_model_and_processor

	# Mock data loading - replace with your actual dataset
	raw_train_data = [None]
	train_dataset = MathReasoningDataset(raw_train_data, processor)

	training_arguments = TrainingArguments(
		output_dir="./checkpoint/qwen_math_reasoning"
		per_device_train_batch_size=2,
		per_device_eval_batch_size=2,
		gradient_accumulation_steps=8,
		learning_rate=2e-5,
		bf16=True,
		logging_steps=10,
		save_strategy="steps",
		report_to="wandb"
	)

	trainer = Trainer(
		model=model,
		args=training_arguments,
		train_dataset=train_dataset,
		data_collator=None,
		compute_metrics=build_compute_metrics_fn(processor, max_optimal_length=600),
		processor_logits_for_metrics=None
	)

	trainer.train()

if __name__ == "__main__":
	run_training()