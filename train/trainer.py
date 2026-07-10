import os

from transformers import Trainer, TrainingArguments

from data.load_data import load_math_dataset
from .dataset import MathReasoningDataset
from .model_builder import build_model_and_processor
from .metrics import build_compute_metrics_fn


def run_training():
    model, processor = build_model_and_processor()

    train_ds, val_ds, _ = load_math_dataset()
    train_dataset = MathReasoningDataset(train_ds, processor)
    eval_dataset = MathReasoningDataset(val_ds, processor)

    training_arguments = TrainingArguments(
        output_dir="./checkpoint/qwen_math_reasoning",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        report_to=os.getenv("REPORT_TO", "none")
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics_fn(processor, max_optimal_length=600)
    )

    trainer.train()


if __name__ == "__main__":
    run_training()
