import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import load_dataset


def load_env_file(env_path: Path) -> None:
	"""Load .env key-value pairs into process environment without overriding existing vars."""
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


def normalize_whitespace(text: str) -> str:
	return re.sub(r"\s+", " ", text).strip()


def normalize_punctuation(text: str) -> str:
	# Keep medically relevant punctuation while removing noisy symbols.
	text = re.sub(r"[^\w\s\.,:;\-\+/()%]", " ", text)
	text = re.sub(r"([\.,:;\-\+/()%])\1+", r"\1", text)
	text = re.sub(r"\s+([\.,:;\)])", r"\1", text)
	text = re.sub(r"([\(])\s+", r"\1", text)
	return normalize_whitespace(text)


ABBREVIATION_MAP: Dict[str, str] = {
	"pt": "patient",
	"hx": "history",
	"dx": "diagnosis",
	"tx": "treatment",
	"rx": "prescription",
	"sx": "symptoms",
	"c/o": "complains of",
	"s/p": "status post",
	"h/o": "history of",
	"w/": "with",
	"w/o": "without",
}


def expand_abbreviations(text: str) -> str:
	normalized = text
	# Handle slash-based abbreviations first.
	for abbr, expanded in ABBREVIATION_MAP.items():
		if "/" in abbr:
			pattern = re.compile(rf"(?<!\w){re.escape(abbr)}(?!\w)", flags=re.IGNORECASE)
			normalized = pattern.sub(expanded, normalized)

	for abbr, expanded in ABBREVIATION_MAP.items():
		if "/" in abbr:
			continue
		pattern = re.compile(rf"\b{re.escape(abbr)}\b", flags=re.IGNORECASE)
		normalized = pattern.sub(expanded, normalized)

	return normalized


LAB_NAMES = [
	"k\\+",
	"na",
	"cl",
	"hgb",
	"wbc",
	"plt",
	"cr",
	"creatinine",
	"bun",
	"glucose",
	"a1c",
	"ast",
	"alt",
	"inr",
]

LAB_PATTERN = re.compile(
	rf"\b(?P<name>{'|'.join(LAB_NAMES)})\b\s*[:=]?\s*(?P<value>\d+(?:\.\d+)?)",
	flags=re.IGNORECASE,
)


def normalize_lab_values(text: str) -> str:
	def _replace(match: re.Match) -> str:
		name = match.group("name").upper()
		value = match.group("value")
		return f"LAB(name={name},value={value})"

	return LAB_PATTERN.sub(_replace, text)


def apply_casing_policy(text: str) -> str:
	# A consistent lowercase policy improves robustness across mixed clinical notation.
	return text.lower()


def truncate_text(text: str, max_chars: int) -> Tuple[str, bool]:
	if max_chars <= 0 or len(text) <= max_chars:
		return text, False
	truncated = text[:max_chars].rstrip()
	return truncated + " ... [truncated]", True


def clean_clinical_text(text: str, max_chars: int) -> Tuple[str, bool]:
	cleaned = normalize_whitespace(text)
	cleaned = expand_abbreviations(cleaned)
	cleaned = normalize_lab_values(cleaned)
	cleaned = normalize_punctuation(cleaned)
	cleaned = apply_casing_policy(cleaned)
	cleaned, was_truncated = truncate_text(cleaned, max_chars=max_chars)
	return cleaned, was_truncated


def extract_summary(summary_value) -> str:
	if isinstance(summary_value, list):
		return " ".join(str(x) for x in summary_value if x is not None)
	return str(summary_value)


def maybe_limit_split(dataset_split, max_samples: int, seed: int):
	if max_samples <= 0:
		return dataset_split
	size = min(max_samples, len(dataset_split))
	shuffled = dataset_split.shuffle(seed=seed)
	return shuffled.select(range(size))


@dataclass
class SplitStats:
	split_name: str
	rows_written: int
	truncated_inputs: int
	truncated_summaries: int
	avg_input_chars: float
	avg_summary_chars: float
	output_file: str


def process_split(
	split_name: str,
	split_data,
	output_file: Path,
	raw_sample_file: Path,
	max_input_chars: int,
	max_summary_chars: int,
	sample_capture_count: int,
) -> Tuple[SplitStats, List[Dict]]:
	output_file.parent.mkdir(parents=True, exist_ok=True)
	raw_sample_file.parent.mkdir(parents=True, exist_ok=True)

	truncated_inputs = 0
	truncated_summaries = 0
	total_input_chars = 0
	total_summary_chars = 0
	rows_written = 0
	samples: List[Dict] = []

	with output_file.open("w", encoding="utf-8") as out_f, raw_sample_file.open(
		"w", encoding="utf-8"
	) as raw_f:
		for idx, row in enumerate(split_data):
			source_text = str(row.get("article", ""))
			target_text = extract_summary(row.get("abstract", ""))

			cleaned_source, source_truncated = clean_clinical_text(source_text, max_input_chars)
			cleaned_target, target_truncated = clean_clinical_text(target_text, max_summary_chars)

			if source_truncated:
				truncated_inputs += 1
			if target_truncated:
				truncated_summaries += 1

			total_input_chars += len(cleaned_source)
			total_summary_chars += len(cleaned_target)

			processed = {
				"id": f"{split_name}-{idx}",
				"text": cleaned_source,
				"summary": cleaned_target,
			}
			out_f.write(json.dumps(processed, ensure_ascii=True) + "\n")

			if idx < sample_capture_count:
				trace = {
					"id": processed["id"],
					"raw_text_preview": source_text[:500],
					"cleaned_text_preview": cleaned_source[:500],
					"raw_summary_preview": target_text[:300],
					"cleaned_summary_preview": cleaned_target[:300],
				}
				samples.append(trace)
				raw_f.write(json.dumps(trace, ensure_ascii=True) + "\n")

			rows_written += 1

	avg_input = (total_input_chars / rows_written) if rows_written else 0.0
	avg_summary = (total_summary_chars / rows_written) if rows_written else 0.0

	stats = SplitStats(
		split_name=split_name,
		rows_written=rows_written,
		truncated_inputs=truncated_inputs,
		truncated_summaries=truncated_summaries,
		avg_input_chars=round(avg_input, 2),
		avg_summary_chars=round(avg_summary, 2),
		output_file=str(output_file.as_posix()),
	)
	return stats, samples


def main() -> None:
	repo_root = Path(__file__).resolve().parents[1]
	load_env_file(repo_root / ".env")

	random.seed(get_int_env("SEED", 42))

	dataset_name = os.getenv("DATASET_NAME", "ccdv/pubmed-summarization")
	train_split_name = os.getenv("TRAIN_SPLIT", "train")
	val_split_name = os.getenv("VALIDATION_SPLIT", "validation")
	test_split_name = os.getenv("TEST_SPLIT", "test")

	max_train_samples = get_int_env("MAX_TRAIN_SAMPLES", 500)
	max_val_samples = get_int_env("MAX_EVAL_SAMPLES", 100)
	max_test_samples = get_int_env("MAX_TEST_SAMPLES", 100)
	max_input_chars = get_int_env("MAX_INPUT_CHARS", 7000)
	max_summary_chars = get_int_env("MAX_SUMMARY_CHARS", 1200)
	seed = get_int_env("SEED", 42)
	sample_capture_count = get_int_env("TRACEABILITY_SAMPLE_COUNT", 3)

	raw_dir = repo_root / os.getenv("RAW_DATA_DIR", "data/raw")
	processed_dir = repo_root / os.getenv("PROCESSED_DATA_DIR", "data/processed")
	raw_dir.mkdir(parents=True, exist_ok=True)
	processed_dir.mkdir(parents=True, exist_ok=True)

	train_ds = load_dataset(dataset_name, split=train_split_name)
	val_ds = load_dataset(dataset_name, split=val_split_name)
	test_ds = load_dataset(dataset_name, split=test_split_name)

	train_ds = maybe_limit_split(train_ds, max_train_samples, seed)
	val_ds = maybe_limit_split(val_ds, max_val_samples, seed)
	test_ds = maybe_limit_split(test_ds, max_test_samples, seed)

	split_configs = [
		("train", train_ds, processed_dir / "train.jsonl", raw_dir / "train_raw_samples.jsonl"),
		(
			"validation",
			val_ds,
			processed_dir / "validation.jsonl",
			raw_dir / "validation_raw_samples.jsonl",
		),
		("test", test_ds, processed_dir / "test.jsonl", raw_dir / "test_raw_samples.jsonl"),
	]

	all_stats: Dict[str, Dict] = {}
	all_samples: Dict[str, List[Dict]] = {}

	for split_name, split_ds, output_file, raw_sample_file in split_configs:
		stats, samples = process_split(
			split_name=split_name,
			split_data=split_ds,
			output_file=output_file,
			raw_sample_file=raw_sample_file,
			max_input_chars=max_input_chars,
			max_summary_chars=max_summary_chars,
			sample_capture_count=sample_capture_count,
		)
		all_stats[split_name] = {
			"rows_written": stats.rows_written,
			"truncated_inputs": stats.truncated_inputs,
			"truncated_summaries": stats.truncated_summaries,
			"avg_input_chars": stats.avg_input_chars,
			"avg_summary_chars": stats.avg_summary_chars,
			"output_file": stats.output_file,
		}
		all_samples[split_name] = samples

	metadata = {
		"dataset_name": dataset_name,
		"seed": seed,
		"max_train_samples": max_train_samples,
		"max_eval_samples": max_val_samples,
		"max_test_samples": max_test_samples,
		"max_input_chars": max_input_chars,
		"max_summary_chars": max_summary_chars,
	}

	(processed_dir / "split_stats.json").write_text(
		json.dumps({"metadata": metadata, "splits": all_stats}, indent=2), encoding="utf-8"
	)
	(processed_dir / "sample_rows.json").write_text(json.dumps(all_samples, indent=2), encoding="utf-8")

	print("Preprocessing completed.")
	print(f"Processed files written to: {processed_dir.as_posix()}")


if __name__ == "__main__":
	main()
