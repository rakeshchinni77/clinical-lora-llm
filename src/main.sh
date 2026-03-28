#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f ".env" ]]; then
	set -a
	# Load .env robustly even when file has Windows CRLF line endings.
	# shellcheck disable=SC1091
	source <(sed 's/\r$//' .env)
	set +a
fi

RAW_DATA_DIR="${RAW_DATA_DIR:-data/raw}"
PROCESSED_DATA_DIR="${PROCESSED_DATA_DIR:-data/processed}"
FINAL_ADAPTER_DIR="${FINAL_ADAPTER_DIR:-output/final_adapter}"
RESULTS_DIR="${RESULTS_DIR:-results}"
LOG_DIR="${LOG_DIR:-results/logs}"
UNSTABLE_LR="${UNSTABLE_LEARNING_RATE:-1e-2}"
STABLE_LR="${STABLE_LEARNING_RATE:-2e-4}"

mkdir -p "$RAW_DATA_DIR" "$PROCESSED_DATA_DIR" "$FINAL_ADAPTER_DIR" "$RESULTS_DIR" "$LOG_DIR"

log() {
	printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
}

run_step() {
	local label="$1"
	shift
	log "$label"
	"$@"
}

require_file() {
	local path="$1"
	if [[ ! -s "$path" ]]; then
		log "Missing or empty required artifact: $path"
		exit 1
	fi
}

run_step "Step 1/5: Preprocess clinical dataset" \
	python src/preprocess.py

run_step "Step 2/5: Run intentionally unstable training" \
	/bin/bash -lc "python src/train.py --run-type unstable --learning-rate ${UNSTABLE_LR} 2>&1 | tee ${LOG_DIR}/unstable.log"

run_step "Step 3/5: Run stable LoRA training" \
	/bin/bash -lc "python src/train.py --run-type stable --learning-rate ${STABLE_LR} 2>&1 | tee ${LOG_DIR}/stable.log"

run_step "Step 4/5: Evaluate base and fine-tuned models" \
	python src/evaluate.py

log "Step 5/5: Validate expected report artifacts"
require_file "${RESULTS_DIR}/training_analysis.json"
require_file "${RESULTS_DIR}/evaluation_metrics.json"

log "Pipeline completed successfully."
