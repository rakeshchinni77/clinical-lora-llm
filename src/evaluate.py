"""
Evaluation script for clinical LoRA fine-tuning:
- ROUGE-1/2/L metrics (base vs fine-tuned models)
- Custom Clinical Entity Recall metric
- Dual-model comparison report saved to results/evaluation_metrics.json
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Set

import torch
from peft import PeftModel
from rouge_score import rouge_scorer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed


def load_env_file(env_path: Path) -> None:
	"""Load .env file into os.environ if it exists."""
	import os

	if not env_path.exists():
		return

	for raw_line in env_path.read_text(encoding="utf-8").splitlines():
		line = raw_line.strip()
		if not line or line.startswith("#") or "=" not in line:
			continue
		key, value = line.split("=", 1)
		key = key.strip()
		value = value.strip().strip('"').strip("'")
		if key and key not in os.environ:
			os.environ[key] = value


def get_str_env(name: str, default: str) -> str:
	import os

	return os.getenv(name, default)


CLINICAL_ENTITIES = {
	"disease", "disorder", "syndrome", "infection", "illness", "cancer",
	"diabetes", "hypertension", "cardiovascular", "stroke", "heart attack",
	"pneumonia", "sepsis", "trauma", "fracture", "arthritis", "pain",
	"anxiety", "depression", "stress", "premature", "nicu", "icu",
	"fever", "inflammation", "wound", "injury",
	"treatment", "therapy", "intervention", "surgery", "operation", "transplant",
	"medication", "drug", "antibiotics", "vaccine", "chemotherapy", "radiation",
	"rehabilitation", "physiotherapy", "counseling", "education", "training",
	"management", "care", "program", "protocol",
	"imaging", "scan", "mri", "ct", "ultrasound", "endoscopy", "biopsy",
	"examination", "test", "assessment", "evaluation", "diagnosis", "monitoring",
	"heart", "lung", "liver", "kidney", "brain", "spine", "joint", "muscle",
	"blood", "respiratory", "gastrointestinal", "neurological", "immune", "endocrine", "system",
	"patient", "infant", "neonate", "pediatric", "adult", "elderly", "mother",
	"child", "children", "women", "men", "population", "cohort",
	"outcome", "mortality", "morbidity", "recovery", "remission", "relapse",
	"complication", "adverse", "benefit", "efficacy", "safety", "quality", "survival", "prognosis",
}


def extract_clinical_entities(text: str) -> Set[str]:
	text_lower = text.lower()
	text_words = re.findall(r"\b[a-z_-]+\b", text_lower)
	text_set = set(text_words)
	found_entities = text_set & CLINICAL_ENTITIES

	clinical_terms = [
		r"\bneonatal intensive care\b",
		r"\bintensive care unit\b",
		r"\bclinical trial\b",
		r"\brandomized controlled\b",
		r"\bside effect\b",
		r"\badverse event\b",
		r"\bquality of life\b",
		r"\bparent empowerment\b",
		r"\bcope program\b",
		r"\b(maternal|paternal|parental)\s+(stress|anxiety|depression)\b",
		r"\b(premature|preterm)\s+(infant|birth)\b",
	]

	for pattern in clinical_terms:
		matches = re.findall(pattern, text_lower)
		for match in matches:
			if isinstance(match, tuple):
				match = " ".join(m for m in match if m)
			found_entities.add(match.lower().replace(" ", "_"))

	return found_entities


def clinical_entity_recall(reference: str, generated: str) -> float:
	ref_entities = extract_clinical_entities(reference)
	gen_entities = extract_clinical_entities(generated)
	if not ref_entities:
		return 1.0
	return len(ref_entities & gen_entities) / len(ref_entities)


def load_base_model(model_name: str, device: str):
	print(f"[Base Model] Loading {model_name}...")
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = AutoModelForSeq2SeqLM.from_pretrained(
		model_name,
		torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
		device_map="auto" if device == "cuda" else None,
	)
	if device == "cpu":
		model = model.to(device)
	model.eval()
	return model, tokenizer


def load_finetuned_model(base_model_name: str, adapter_dir: str, device: str):
	print(f"[Fine-tuned Model] Loading {base_model_name} with adapter from {adapter_dir}...")
	tokenizer = AutoTokenizer.from_pretrained(base_model_name)
	base_model = AutoModelForSeq2SeqLM.from_pretrained(
		base_model_name,
		torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
		device_map="auto" if device == "cuda" else None,
	)
	if device == "cpu":
		base_model = base_model.to(device)
	model = PeftModel.from_pretrained(base_model, adapter_dir)
	model.eval()
	return model, tokenizer


def generate_summary(model, tokenizer, text: str, device: str, max_length: int = 150) -> str:
	inputs = tokenizer(text, max_length=512, truncation=True, return_tensors="pt")
	if device == "cuda":
		inputs = {k: v.cuda() for k, v in inputs.items()}
	with torch.no_grad():
		outputs = model.generate(**inputs, max_length=max_length, num_beams=4, temperature=1.0)
	return tokenizer.decode(outputs[0], skip_special_tokens=True)


def evaluate_models(base_model, finetuned_model, tokenizer, test_examples: List[Dict[str, str]], device: str, max_samples: int = -1) -> Dict[str, Any]:
	scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
	if max_samples > 0:
		test_examples = test_examples[:max_samples]

	base_summaries, finetuned_summaries, references = [], [], []
	base_entity_recalls, finetuned_entity_recalls = [], []
	sample_predictions = []

	print(f"\n[Inference] Processing {len(test_examples)} examples...")
	for idx, example in enumerate(test_examples):
		if (idx + 1) % max(1, len(test_examples) // 10) == 0:
			print(f"  Progress: {idx + 1}/{len(test_examples)}")
		text = example["text"]
		reference = example["summary"]
		references.append(reference)

		base_summary = generate_summary(base_model, tokenizer, text, device)
		finetuned_summary = generate_summary(finetuned_model, tokenizer, text, device)
		base_summaries.append(base_summary)
		finetuned_summaries.append(finetuned_summary)

		base_recall = clinical_entity_recall(reference, base_summary)
		finetuned_recall = clinical_entity_recall(reference, finetuned_summary)
		base_entity_recalls.append(base_recall)
		finetuned_entity_recalls.append(finetuned_recall)

		if idx < 3:
			sample_predictions.append(
				{
					"example_id": example.get("id", f"test-{idx}"),
					"reference": reference,
					"base_model_summary": base_summary,
					"finetuned_model_summary": finetuned_summary,
					"base_entity_recall": float(base_recall),
					"finetuned_entity_recall": float(finetuned_recall),
				}
			)

	print("\n[ROUGE] Computing ROUGE metrics...")
	base_rouge1 = base_rouge2 = base_rougeL = 0.0
	ft_rouge1 = ft_rouge2 = ft_rougeL = 0.0
	for reference, base_pred, ft_pred in zip(references, base_summaries, finetuned_summaries):
		base_scores = scorer.score(reference, base_pred)
		ft_scores = scorer.score(reference, ft_pred)
		base_rouge1 += base_scores["rouge1"].fmeasure
		base_rouge2 += base_scores["rouge2"].fmeasure
		base_rougeL += base_scores["rougeL"].fmeasure
		ft_rouge1 += ft_scores["rouge1"].fmeasure
		ft_rouge2 += ft_scores["rouge2"].fmeasure
		ft_rougeL += ft_scores["rougeL"].fmeasure

	count = max(1, len(references))
	base_rouge = {"rouge1": base_rouge1 / count, "rouge2": base_rouge2 / count, "rougeL": base_rougeL / count}
	finetuned_rouge = {"rouge1": ft_rouge1 / count, "rouge2": ft_rouge2 / count, "rougeL": ft_rougeL / count}

	base_avg_recall = sum(base_entity_recalls) / len(base_entity_recalls) if base_entity_recalls else 0.0
	finetuned_avg_recall = sum(finetuned_entity_recalls) / len(finetuned_entity_recalls) if finetuned_entity_recalls else 0.0

	results = {
		"base_model_metrics": {
			"ROUGE-1": float(base_rouge["rouge1"]),
			"ROUGE-2": float(base_rouge["rouge2"]),
			"ROUGE-L": float(base_rouge["rougeL"]),
			"clinical_entity_recall": float(base_avg_recall),
		},
		"fine_tuned_model_metrics": {
			"ROUGE-1": float(finetuned_rouge["rouge1"]),
			"ROUGE-2": float(finetuned_rouge["rouge2"]),
			"ROUGE-L": float(finetuned_rouge["rougeL"]),
			"clinical_entity_recall": float(finetuned_avg_recall),
		},
		"comparison": {
			"rouge1_improvement": float(finetuned_rouge["rouge1"] - base_rouge["rouge1"]),
			"rouge2_improvement": float(finetuned_rouge["rouge2"] - base_rouge["rouge2"]),
			"rougeL_improvement": float(finetuned_rouge["rougeL"] - base_rouge["rougeL"]),
			"entity_recall_improvement": float(finetuned_avg_recall - base_avg_recall),
		},
		"sample_predictions": sample_predictions,
	}
	results["base_model"] = results["base_model_metrics"]
	results["finetuned_model"] = results["fine_tuned_model_metrics"]
	return results


def main():
	parser = argparse.ArgumentParser(description="Evaluate clinical LoRA fine-tuning")
	parser.add_argument("--max-samples", type=int, default=-1, help="Max test samples to evaluate (-1 for all)")
	parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda or cpu)")
	args = parser.parse_args()

	load_env_file(Path(".env"))
	base_model_name = get_str_env("BASE_MODEL_NAME", "google/flan-t5-base")
	adapter_dir = get_str_env("FINAL_ADAPTER_DIR", "output/final_adapter")
	test_data_path = get_str_env("TEST_JSONL_PATH", "data/processed/test.jsonl")
	output_json_path = get_str_env("EVAL_OUTPUT_JSON", "results/evaluation_metrics.json")

	device = args.device
	print(f"[Config] Using device: {device}")
	print(f"[Config] Base model: {base_model_name}")
	print(f"[Config] Adapter: {adapter_dir}")
	print(f"[Config] Test data: {test_data_path}")

	set_seed(42)
	print(f"\n[Loading] Test data from {test_data_path}...")
	test_examples = []
	with open(test_data_path, "r", encoding="utf-8") as f:
		for line in f:
			test_examples.append(json.loads(line))
	print(f"[Loading] Loaded {len(test_examples)} test examples")

	print("\n[Loading Models]")
	base_model, tokenizer = load_base_model(base_model_name, device)
	finetuned_model, _ = load_finetuned_model(base_model_name, adapter_dir, device)

	print("\n[Evaluation]")
	results = evaluate_models(
		base_model,
		finetuned_model,
		tokenizer,
		test_examples,
		device,
		max_samples=args.max_samples if args.max_samples > 0 else len(test_examples),
	)

	output_path = Path(output_json_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with open(output_path, "w", encoding="utf-8") as f:
		json.dump(results, f, indent=2)

	print(f"\n[Success] Results saved to {output_json_path}")
	print("\n" + "=" * 70)
	print("EVALUATION SUMMARY")
	print("=" * 70)
	print("\nBase Model Metrics:")
	for metric, value in results["base_model_metrics"].items():
		print(f"  {metric}: {value:.4f}")
	print("\nFine-tuned Model Metrics:")
	for metric, value in results["fine_tuned_model_metrics"].items():
		print(f"  {metric}: {value:.4f}")
	print("\nImprovement (Fine-tuned vs Base):")
	for metric, value in results["comparison"].items():
		sign = "+" if value > 0 else ""
		print(f"  {metric}: {sign}{value:.4f}")


if __name__ == "__main__":
	main()
