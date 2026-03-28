import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
	AutoModelForSeq2SeqLM,
	AutoTokenizer,
	DataCollatorForSeq2Seq,
	Seq2SeqTrainer,
	Seq2SeqTrainingArguments,
	set_seed,
)


def load_env_file(env_path: Path) -> None:
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


def get_int_env(name: str, default: int) -> int:
	value = os.getenv(name)
	if value is None:
		return default
	try:
		return int(value)
	except ValueError:
		return default


def get_float_env(name: str, default: float) -> float:
	value = os.getenv(name)
	if value is None:
		return default
	try:
		return float(value)
	except ValueError:
		return default


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="LoRA fine-tuning for clinical summarization.")
	parser.add_argument("--run-type", choices=["stable", "unstable"], required=True)
	parser.add_argument("--learning-rate", type=float, default=None)
	parser.add_argument("--force-cpu", action="store_true", help="Force training on CPU.")
	return parser.parse_args()


def resolve_runtime(args: argparse.Namespace) -> Dict[str, Any]:
	repo_root = Path(__file__).resolve().parents[1]
	load_env_file(repo_root / ".env")

	seed = get_int_env("SEED", 42)
	set_seed(seed)

	base_model_name = os.getenv("BASE_MODEL_NAME", "google/flan-t5-base")
	hf_token = os.getenv("HF_TOKEN")

	processed_dir = repo_root / os.getenv("PROCESSED_DATA_DIR", "data/processed")
	output_dir = repo_root / os.getenv("OUTPUT_DIR", "output")
	results_dir = repo_root / os.getenv("RESULTS_DIR", "results")
	final_adapter_dir = repo_root / os.getenv("FINAL_ADAPTER_DIR", "output/final_adapter")

	max_input_length = get_int_env("MAX_INPUT_LENGTH", 512)
	max_target_length = get_int_env("MAX_TARGET_LENGTH", 128)

	lora_rank = get_int_env("LORA_RANK", 16)
	lora_alpha = get_int_env("LORA_ALPHA", 32)
	lora_dropout = get_float_env("LORA_DROPOUT", 0.05)
	lora_target_modules = [
		module.strip()
		for module in os.getenv("LORA_TARGET_MODULES", "q,v").split(",")
		if module.strip()
	]

	num_epochs = get_int_env("NUM_EPOCHS", 1)
	train_batch_size = get_int_env("PER_DEVICE_TRAIN_BATCH_SIZE", 2)
	eval_batch_size = get_int_env("PER_DEVICE_EVAL_BATCH_SIZE", 2)
	grad_accum = get_int_env("GRADIENT_ACCUMULATION_STEPS", 4)
	warmup_ratio = get_float_env("WARMUP_RATIO", 0.05)
	weight_decay = get_float_env("WEIGHT_DECAY", 0.01)
	logging_steps = get_int_env("LOGGING_STEPS", 10)
	save_total_limit = get_int_env("SAVE_TOTAL_LIMIT", 2)

	default_lr = (
		get_float_env("STABLE_LEARNING_RATE", 2e-4)
		if args.run_type == "stable"
		else get_float_env("UNSTABLE_LEARNING_RATE", 1e-2)
	)
	learning_rate = args.learning_rate if args.learning_rate is not None else default_lr

	force_cpu = args.force_cpu or os.getenv("FORCE_CPU", "0") == "1"
	use_cuda = torch.cuda.is_available() and not force_cpu
	device = "cuda" if use_cuda else "cpu"
	use_bf16 = bool(use_cuda and torch.cuda.is_bf16_supported())

	run_output_dir = output_dir / "checkpoints" / args.run_type
	run_output_dir.mkdir(parents=True, exist_ok=True)
	results_dir.mkdir(parents=True, exist_ok=True)
	final_adapter_dir.mkdir(parents=True, exist_ok=True)

	return {
		"repo_root": repo_root,
		"seed": seed,
		"base_model_name": base_model_name,
		"hf_token": hf_token,
		"processed_dir": processed_dir,
		"output_dir": output_dir,
		"results_dir": results_dir,
		"final_adapter_dir": final_adapter_dir,
		"max_input_length": max_input_length,
		"max_target_length": max_target_length,
		"lora_rank": lora_rank,
		"lora_alpha": lora_alpha,
		"lora_dropout": lora_dropout,
		"lora_target_modules": lora_target_modules,
		"num_epochs": num_epochs,
		"train_batch_size": train_batch_size,
		"eval_batch_size": eval_batch_size,
		"grad_accum": grad_accum,
		"warmup_ratio": warmup_ratio,
		"weight_decay": weight_decay,
		"logging_steps": logging_steps,
		"save_total_limit": save_total_limit,
		"learning_rate": learning_rate,
		"device": device,
		"use_bf16": use_bf16,
		"run_output_dir": run_output_dir,
	}


def load_processed_splits(processed_dir: Path):
	train_path = processed_dir / "train.jsonl"
	val_path = processed_dir / "validation.jsonl"

	if not train_path.exists() or not val_path.exists():
		raise FileNotFoundError(
			"Missing processed datasets. Expected data/processed/train.jsonl and validation.jsonl."
		)

	return load_dataset(
		"json",
		data_files={
			"train": str(train_path),
			"validation": str(val_path),
		},
	)


def tokenize_datasets(dataset_dict, tokenizer, max_input_length: int, max_target_length: int):
	def preprocess_batch(batch):
		model_inputs = tokenizer(
			batch["text"],
			max_length=max_input_length,
			truncation=True,
			padding=False,
		)
		labels = tokenizer(
			text_target=batch["summary"],
			max_length=max_target_length,
			truncation=True,
			padding=False,
		)
		model_inputs["labels"] = labels["input_ids"]
		return model_inputs

	tokenized = dataset_dict.map(
		preprocess_batch,
		batched=True,
		remove_columns=dataset_dict["train"].column_names,
		desc="Tokenizing processed clinical data",
	)
	return tokenized


def safe_metric(value: Any) -> Any:
	if value is None:
		return "NaN"
	if isinstance(value, (int, float)):
		if math.isnan(value) or math.isinf(value):
			return "NaN"
		return float(round(value, 6))
	return value


def init_or_load_training_report(report_path: Path) -> Dict[str, Any]:
	default_report = {
		"stable_run": {
			"log_path": "results/logs/stable.log",
			"final_metrics": {"train_loss": "NaN", "eval_loss": "NaN"},
		},
		"unstable_run": {
			"log_path": "results/logs/unstable.log",
			"final_metrics": {"train_loss": "NaN", "eval_loss": "NaN"},
		},
	}
	if not report_path.exists():
		return default_report

	try:
		existing = json.loads(report_path.read_text(encoding="utf-8"))
		if isinstance(existing, dict):
			default_report.update(existing)
	except json.JSONDecodeError:
		pass
	return default_report


def update_training_report(
	report_path: Path,
	run_type: str,
	train_loss: Any,
	eval_loss: Any,
) -> None:
	report = init_or_load_training_report(report_path)
	run_key = f"{run_type}_run"
	report[run_key] = {
		"log_path": f"results/logs/{run_type}.log",
		"final_metrics": {
			"train_loss": safe_metric(train_loss),
			"eval_loss": safe_metric(eval_loss),
		},
	}
	report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def train_once(args: argparse.Namespace, cfg: Dict[str, Any]) -> Tuple[Any, Any]:
	token = cfg["hf_token"] if cfg["hf_token"] else None
	tokenizer = AutoTokenizer.from_pretrained(cfg["base_model_name"], token=token)

	model = AutoModelForSeq2SeqLM.from_pretrained(cfg["base_model_name"], token=token)
	if cfg["device"] == "cuda":
		model = model.to("cuda")

	lora_config = LoraConfig(
		r=cfg["lora_rank"],
		lora_alpha=cfg["lora_alpha"],
		target_modules=cfg["lora_target_modules"],
		lora_dropout=cfg["lora_dropout"],
		bias="none",
		task_type=TaskType.SEQ_2_SEQ_LM,
	)
	model = get_peft_model(model, lora_config)
	model.print_trainable_parameters()

	datasets_dict = load_processed_splits(cfg["processed_dir"])
	tokenized = tokenize_datasets(
		datasets_dict,
		tokenizer,
		max_input_length=cfg["max_input_length"],
		max_target_length=cfg["max_target_length"],
	)

	data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

	max_steps = -1
	if args.run_type == "unstable":
		# Keep unstable experiment short; intentionally high LR is the instability driver.
		max_steps = get_int_env("UNSTABLE_MAX_STEPS", 25)
	else:
		stable_cap = get_int_env("STABLE_MAX_STEPS", -1)
		if stable_cap > 0:
			max_steps = stable_cap

	training_args = Seq2SeqTrainingArguments(
		output_dir=cfg["output_dir"],
		num_train_epochs=cfg["num_epochs"],
		max_steps=max_steps,
		learning_rate=cfg["learning_rate"],
		per_device_train_batch_size=cfg["train_batch_size"],
		per_device_eval_batch_size=cfg["eval_batch_size"],
		gradient_accumulation_steps=cfg["grad_accum"],
		warmup_ratio=cfg["warmup_ratio"],
		weight_decay=cfg["weight_decay"],
		logging_steps=cfg["logging_steps"],
		eval_strategy="epoch",
		save_strategy="epoch",
		save_total_limit=cfg["save_total_limit"],
		predict_with_generate=False,
		bf16=cfg["use_bf16"],
		fp16=False,
		report_to=[],
		dataloader_pin_memory=False,
		remove_unused_columns=True,
	)

	trainer = Seq2SeqTrainer(
		model=model,
		args=training_args,
		train_dataset=tokenized["train"],
		eval_dataset=tokenized["validation"],
		tokenizer=tokenizer,
		data_collator=data_collator,
	)

	train_result = trainer.train()
	eval_metrics = trainer.evaluate()

	trainer.save_state()
	trainer.save_metrics("train", train_result.metrics)
	trainer.save_metrics("eval", eval_metrics)

	if args.run_type == "stable":
		model.save_pretrained(str(cfg["final_adapter_dir"]))
		tokenizer.save_pretrained(str(cfg["final_adapter_dir"]))

	train_loss = train_result.metrics.get("train_loss", train_result.training_loss)
	eval_loss = eval_metrics.get("eval_loss")
	return train_loss, eval_loss


def main() -> None:
	args = parse_args()
	cfg = resolve_runtime(args)

	train_loss = "NaN"
	eval_loss = "NaN"
	report_path = cfg["results_dir"] / "training_analysis.json"

	try:
		train_loss, eval_loss = train_once(args, cfg)
	except Exception as exc:
		print(f"Training run failed for mode={args.run_type}: {exc}")
	finally:
		update_training_report(
			report_path=report_path,
			run_type=args.run_type,
			train_loss=train_loss,
			eval_loss=eval_loss,
		)
		print(f"Updated training report: {report_path.as_posix()}")


if __name__ == "__main__":
	main()
