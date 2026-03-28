# clinical-lora-llm

Domain-Adaptive Clinical LLM using LoRA fine-tuning, containerized pipeline orchestration, evaluation, and quality documentation.

## Hardware Note

“Due to hardware constraints, I used FLAN-T5-base. The pipeline is fully compatible with larger models like Mistral-7B when deployed on GPU infrastructure.”

## Architecture

This project executes a 5-step pipeline through `src/main.sh`:

1. Preprocess clinical summarization data into JSONL files.
2. Run unstable LoRA training (high learning rate, stress test).
3. Run stable LoRA training (production candidate adapter).
4. Evaluate base vs fine-tuned model (ROUGE-1/2/L + clinical entity recall).
5. Validate mandatory artifacts and schema output files.

Core runtime components:

- `src/preprocess.py`: data loading, cleaning, and processed split generation.
- `src/train.py`: LoRA fine-tuning (stable and unstable paths).
- `src/evaluate.py`: expected evaluation entrypoint.
- `docker-compose.yml`: CPU default service and GPU profile service.
- `src/main.sh`: orchestration and final artifact validation.

## Environment Setup

### Option A: Local Python (.venv)

Prerequisites:

- Python 3.10+
- Git

Setup:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Configure environment variables:

```powershell
Copy-Item .env.example .env
```

Run full pipeline locally:

```powershell
bash src/main.sh
```

### Option B: Docker (recommended)

Prerequisites:

- Docker Desktop
- Docker Compose

Create env file if missing:

```powershell
Copy-Item .env.example .env
```

## One-Command Run Path

Run the full pipeline with one command:

```bash
docker-compose up
```

Alternative modern CLI equivalent:

```bash
docker compose up
```

Force image rebuild when needed:

```bash
docker compose up --build
```

GPU profile run:

```bash
docker compose --profile gpu up --build
```

## Outputs

Expected output directories:

- `data/processed/`
- `output/final_adapter/`
- `results/`
- `results/logs/`
- `docs/`

Key files:

- `data/processed/train.jsonl`
- `data/processed/validation.jsonl`
- `data/processed/test.jsonl`
- `output/final_adapter/adapter_model.safetensors`
- `output/final_adapter/adapter_config.json`
- `results/training_analysis.json`
- `results/evaluation_metrics.json`
- `results/hallucination_analysis.md`
- `docs/lora_config.md`

## Schema Contracts

### training_analysis.json (required keys)

Top-level keys:

- `stable_run`
- `unstable_run`

Required nested keys per run:

- `log_path`
- `final_metrics.train_loss`
- `final_metrics.eval_loss`

### evaluation_metrics.json (required keys)

Top-level keys:

- `base_model_metrics`
- `fine_tuned_model_metrics`
- `comparison`
- `sample_predictions`

Required metrics under `base_model_metrics` and `fine_tuned_model_metrics`:

- `ROUGE-1`
- `ROUGE-2`
- `ROUGE-L`
- `clinical_entity_recall`

Required comparison keys:

- `rouge1_improvement`
- `rouge2_improvement`
- `rougeL_improvement`
- `entity_recall_improvement`

## Validation Checklist

Use this checklist after each full run.

Artifact checks:

- [ ] `data/processed/train.jsonl` exists and is non-empty.
- [ ] `data/processed/validation.jsonl` exists and is non-empty.
- [ ] `data/processed/test.jsonl` exists and is non-empty.
- [ ] `output/final_adapter/adapter_model.safetensors` exists.
- [ ] `output/final_adapter/adapter_config.json` exists.
- [ ] `results/logs/unstable.log` exists.
- [ ] `results/logs/stable.log` exists.
- [ ] `results/training_analysis.json` exists and is valid JSON.
- [ ] `results/evaluation_metrics.json` exists and is valid JSON.
- [ ] `results/hallucination_analysis.md` exists.
- [ ] `docs/lora_config.md` exists.

Schema checks:

- [ ] `results/training_analysis.json` includes `stable_run` and `unstable_run` with `final_metrics.train_loss` and `final_metrics.eval_loss`.
- [ ] `results/evaluation_metrics.json` includes `base_model_metrics`, `fine_tuned_model_metrics`, `comparison`, and `sample_predictions`.
- [ ] `results/evaluation_metrics.json` includes ROUGE-1/2/L and clinical_entity_recall for both model metric sections.

Example PowerShell schema probe:

```powershell
$t = Get-Content results/training_analysis.json -Raw | ConvertFrom-Json
$e = Get-Content results/evaluation_metrics.json -Raw | ConvertFrom-Json

$t.stable_run.final_metrics.train_loss
$t.unstable_run.final_metrics.eval_loss
$e.base_model_metrics.'ROUGE-1'
$e.fine_tuned_model_metrics.clinical_entity_recall
$e.comparison.rouge1_improvement
$e.sample_predictions.Count
```

## Troubleshooting

### Circular import with evaluate package

Symptom: `ImportError: cannot import name 'load' from partially initialized module 'evaluate'`.

Fix:

- Evaluation computes ROUGE via `rouge_score` in `src/evaluate.py`.
- Use `src/evaluate.py` as the stable project entrypoint.

### Docker command compatibility

If `docker-compose` is not found, use:

```bash
docker compose up
```

### Slow first run

The first pipeline execution downloads large datasets and model artifacts. This is expected. Later runs are faster due to cache reuse.

### GPU not detected

- Validate Docker GPU runtime installation.
- Use CPU path (`trainer` service) when GPU is unavailable.
- Use GPU profile only when `nvidia-smi` is available in the container runtime.

### Windows line-ending issues in shell

The pipeline loader already strips CRLF in `.env` during startup via `src/main.sh`.

## Phase 7 Documentation Outputs

- Hallucination analysis: `results/hallucination_analysis.md`
- LoRA config: `docs/lora_config.md`
