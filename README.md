# clinical-lora-llm

Domain-Adaptive Clinical LLM using **LoRA fine-tuning**, containerized pipeline orchestration, evaluation, and quality documentation.

---

## Hardware Note

> Due to hardware constraints, I used **FLAN-T5-base**.  
> The pipeline is fully compatible with larger models like **Mistral-7B** when deployed on GPU infrastructure.

---

## System Architecture & Workflow

This project implements a complete **end-to-end pipeline for clinical text summarization** using LoRA-based fine-tuning.

---

### Pipeline Overview

#### 1️.Data Acquisition
- Source: PubMed Summarization dataset (Hugging Face)
- Input: Clinical text + summaries

---

#### 2️.Preprocessing Pipeline (`src/preprocess.py`)
- Abbreviation expansion using regex
- Lab value normalization
- Text cleaning and formatting

**Outputs:**
- `train.jsonl`
- `validation.jsonl`
- `test.jsonl`

---

#### 3️.Model Fine-Tuning (LoRA) (`src/train.py`)
- **Base Model:** `google/flan-t5-base`

**LoRA Configuration:**
```python
r = 16
lora_alpha = 32
target_modules = ["q", "v"]
```

**Training Modes:**
- ❌ Unstable run (high learning rate)
- ✅ Stable run (production-ready model)

---

#### 4️.Evaluation Pipeline (`src/evaluate.py`)
- Base model inference
- Fine-tuned model inference

**Metrics:**
- ROUGE-1 / ROUGE-2 / ROUGE-L
- Clinical Entity Recall (custom metric)

---

#### 5️.Output & Reporting
- `evaluation_metrics.json`
- `training_analysis.json`
- `hallucination_analysis.md`

---

## Workflow Summary

```
Raw Clinical Data
        ↓
Preprocessing Pipeline
        ↓
Cleaned Dataset
        ↓
LoRA Fine-Tuning
        ↓
Fine-Tuned Adapter
        ↓
Evaluation (Base vs Fine-Tuned)
        ↓
Metrics + Reports
```

---

## Project Structure

```
clinical-lora-llm/
│
├── docker-compose.yml
├── Dockerfile
├── .env.example
├── requirements.txt
├── README.md
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── main.sh
│   └── utils.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── output/
│   └── final_adapter/
│       ├── adapter_model.safetensors
│       └── adapter_config.json
│
├── results/
│   ├── evaluation_metrics.json
│   ├── training_analysis.json
│   ├── hallucination_analysis.md
│   └── logs/
│       ├── stable.log
│       └── unstable.log
│
└── docs/
    └── lora_config.md
```

---

## Environment Setup

### Option A: Local (.venv)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
bash src/main.sh
```

---

### Option B: Docker (Recommended)

```powershell
Copy-Item .env.example .env
```

#### One-Command Run

```bash
docker compose up
```

#### Rebuild

```bash
docker compose up --build
```

#### GPU Mode

```bash
docker compose --profile gpu up --build
```

---

## Outputs

### Directories

- `data/processed/`
- `output/final_adapter/`
- `results/`
- `results/logs/`
- `docs/`

---

### Key Files

- `data/processed/train.jsonl`
- `data/processed/validation.jsonl`
- `data/processed/test.jsonl`
- `output/final_adapter/adapter_model.safetensors`
- `output/final_adapter/adapter_config.json`
- `results/training_analysis.json`
- `results/evaluation_metrics.json`
- `results/hallucination_analysis.md`
- `docs/lora_config.md`

---

## Schema Contracts

### training_analysis.json

```json
{
  "stable_run": {
    "log_path": "string",
    "final_metrics": {
      "train_loss": "number",
      "eval_loss": "number"
    }
  },
  "unstable_run": {
    "log_path": "string",
    "final_metrics": {
      "train_loss": "number or NaN",
      "eval_loss": "number or NaN"
    }
  }
}
```

---

### evaluation_metrics.json

```json
{
  "base_model_metrics": {
    "ROUGE-1": "number",
    "ROUGE-2": "number",
    "ROUGE-L": "number",
    "clinical_entity_recall": "number"
  },
  "fine_tuned_model_metrics": {
    "ROUGE-1": "number",
    "ROUGE-2": "number",
    "ROUGE-L": "number",
    "clinical_entity_recall": "number"
  }
}
```

---

## Validation Checklist

### Artifacts

- [ ] Processed dataset exists  
- [ ] Adapter files exist  
- [ ] Logs exist  
- [ ] JSON outputs valid  
- [ ] Markdown reports exist  

### Schema

- [ ] `training_analysis.json` valid  
- [ ] `evaluation_metrics.json` valid  
- [ ] Required keys present  

---

## Troubleshooting

### Circular Import Error

```
ImportError: cannot import name 'load' from evaluate
```

**Fix:**
- Use `src/evaluate.py`
- Avoid naming conflicts with `evaluate` library

---

### Docker Issues

```
docker-compose not found
```

**Fix:**
```bash
docker compose up
```

---

### Slow First Run

- Dataset + model download is expected  
- Subsequent runs are faster  

---

### GPU Not Detected

- Use CPU mode  
- Verify with:
```bash
nvidia-smi
```

---

### Windows CRLF Issue

- Handled automatically in `main.sh`

---

## Phase Outputs

- **Hallucination Analysis →** `results/hallucination_analysis.md`  
- **LoRA Configuration →** `docs/lora_config.md`  

---