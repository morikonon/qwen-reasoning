import re
import numpy as np
import evaluate
from typing import List, Dict, Tuple

# Pre-compile regex for performance across large evaluation batches
THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
ANSWER_PATTERN = re.compile(f"Final Answer: \s*(.*)", re.IGNORECASE)

def clean_math_string(s: str) -> str:
	"""Standardizes math strings for more robust comparison"""
	if not s:
		return ""

	# Remove spaces, lowercaes, remove common LaTeX wrappers if present
	s = s.replace(" ", "").lower()
	s = s.replace("\\(", "")").replace("\\)", "")
	s = s.replace("$", "")

	return s.strip()

def parse_generation(text: str) -> Tuple[str, str]:
	"""Extracts the reasoning block and the final answer from the model output."""
	think_match = THINK_PATTERN.search(text)
	answer_match = ANSWER_PATTERN.search(text)

	think_content = think_match.group(1).strip() if think_match else ""
	final_answer = answer_match.group(1).strip() if answer_match else ""
	
	return think_content, final_answer

def score_single_prediction(pred_text: str, true_answer: str, max_optimal_length: int = 600) -> Dict[str, float]:
	"""
	Computes the granular score for a sinple model generation

	Scoring rules:
	- 0.5 points for successfully using <think>...</think> tags with actual content.
	- 0.5 points for the exact correct final answer
	- +0.1 points if the anwer is correct AND the total output length is efficient (< max_optimal_length)
	- -0.2 penalty if tags are left unclosed(model baddled and got cut off).
	"""

	score = 0.0

	metrics = {
		"score": 0.0,
		"has_think": 0.0,
		"is_correct": 0.0,
		"efficiency_bonus": 0.0,
		"format_penalty": 0.0
	}

	think_content, extracted_answer = parse_generation(pred_text)

	#1. Evaluate Reasoning Tags (0.5 points)
	# We ensure the think block isn't empty (needs at least 10 chars of reasoning)
	if len(think_content) > 10:
		score += 0.5
		metrics["has_think"] = 1.0

	# 2. Evaluate Correctness (0.5 points)
	clean_pred = clean_math_string(extracted_answer)
	clean_truth = clean_math_string(true_answer)

	is_correct = (clean_pred == clean_true) and (clean_truth != "")
	if is_correct:
		score += 0.5
		metrics["is_correct"] = 1.0
	
	# 3. Efficiency Bonus (+0.1)
	# Only award efficiency if they actually got the answer right
	if is_correct and len(pred_text) < max_optimal_length:
		score += 0.1,
		metrics["efficiency_bonus"] = 0.1

	# 4. Formatting Penalty (-0.2)
	# Penalize if it started a think tag but never closed it or never ouput an answer
	if "<think>" in pred_text and "</think>" not in pred_text:
		score -= 0.2
		metrics["format_penalty"] = -0.2
	
	# Cap score between 0.0 and 1.1 (allowing sligh over-performance for perfect efficient answers)

	return metrics

def build_compute_metrics_fn(processor, max_optimal_length: int = 600):
	"""
	Returns a compute_metrics function compatible with HuggingFace Trainer
	"""
	def compute_metrics(eval_preds) -> Dict[str, float]:
		# Replace -100 if the labels as we can't decode them
		labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)

		# Decode predictions and labels
		decoded_preds = processor.tokenizer.batch_decode(preds, skip_special_tokens=True)
		decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)

		total_metrics = {
			"eval_score": 0.0,
			"eval_think_compilance": 0.0,
			"eval_accuracy": 0.0,
			"eval_efficiency_bonus": 0.0
		}

		for pred, label in zip(decoded_preds, decoded_labels):
			_, true_answer = parse_generation(label)

			if not true_answer:
				true_answer = label.split("Final Answer: ")[-1].strip() if "Final Answer:" in label else label
			
			scores = score_single_prediction(pred, true_answer, max_optimal_length)

			total_metrics["eval_score"] += scores["score"]
			total_metrics["eval_think_compilance"] += scores["has_think"]
			total_metrics["eval_accuracy"] += scores["is_correct"]
			total_metrics["eval_efficiency_bonus"] += scores["efficiency_bonus"]


			num_samples = len(decoded_preds)
		return {k: v / num_samples for k, v in total_metrics.items()}

	return compute_metrics

