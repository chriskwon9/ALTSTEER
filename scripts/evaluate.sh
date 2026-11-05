#!/usr/bin/env bash
# Generate responses on every benchmark and score them for one model.
#
# Usage:
#   OPENAI_API_KEY=your-api-key bash scripts/evaluate.sh <model_name>
#
# Needs configs/<model_name>.sh, the vectors in vectors/<model_name>/ and the
# benchmarks from scripts/download_data.sh.
# Scoring calls the GPT-4o API once per evaluated response, which costs money.
# Finished generation and scoring are skipped when the script is run again.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ $# -ne 1 ]]; then
    echo "Usage: bash $0 <model_name>, where configs/<model_name>.sh exists" >&2
    exit 1
fi

MODEL_NAME="$1"
CONFIG="configs/${MODEL_NAME}.sh"
if [[ ! -f "${CONFIG}" ]]; then
    echo "Error: ${CONFIG} not found." >&2
    exit 1
fi
source "${CONFIG}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "Error: OPENAI_API_KEY is not set." >&2
    exit 1
fi

if [[ ! -f data/inputs/math500.jsonl ]]; then
    echo "Error: benchmarks not found. Run bash scripts/download_data.sh first." >&2
    exit 1
fi

VECTOR_DIR="${VECTOR_DIR:-vectors/${MODEL_NAME}}"
for vector in refusal_vector.pt explanation_vector.pt; do
    if [[ ! -f "${VECTOR_DIR}/${vector}" ]]; then
        echo "Error: ${VECTOR_DIR}/${vector} not found. See \"Adding a Model\" in readme.md." >&2
        exit 1
    fi
done

INFERENCE_DIR="results/${MODEL_NAME}/inference"
EVALUATION_DIR="results/${MODEL_NAME}/evaluation"
mkdir -p "${INFERENCE_DIR}" "${EVALUATION_DIR}"
ALTSTEER="${INFERENCE_DIR}/altsteer.json"
REFUSAL_ONLY="${INFERENCE_DIR}/altsteer_refusal_only.json"
EXPLANATION_ONLY="${INFERENCE_DIR}/altsteer_explanation_only.json"
NAIVE="${INFERENCE_DIR}/naive.jsonl"

echo "[1/9] Generating with AltSteer"
python altsteer/src/generate_altsteer.py \
    --model_path "${MODEL_PATH}" \
    --target_layer "${TARGET_LAYER}" \
    --vector_dir "${VECTOR_DIR}" \
    --refusal_strength "${REFUSAL_STRENGTH}" \
    --explanation_strength "${EXPLANATION_STRENGTH}" \
    --output "${ALTSTEER}"

echo "[2/9] Generating with the refusal vector only"
python altsteer/src/generate_refusal_only.py \
    --model_path "${MODEL_PATH}" \
    --target_layer "${TARGET_LAYER}" \
    --vector_dir "${VECTOR_DIR}" \
    --refusal_strength "${REFUSAL_ONLY_STRENGTH}" \
    --output "${REFUSAL_ONLY}"

echo "[3/9] Generating with the explanation vector only"
python altsteer/src/generate_explanation_only.py \
    --model_path "${MODEL_PATH}" \
    --target_layer "${TARGET_LAYER}" \
    --vector_dir "${VECTOR_DIR}" \
    --explanation_strength "${EXPLANATION_ONLY_STRENGTH}" \
    --output "${EXPLANATION_ONLY}"

echo "[4/9] Generating without steering"
python altsteer/src/generate_naive.py \
    --model_path "${MODEL_PATH}" \
    --output "${NAIVE}"

echo "[5/9] Scoring math accuracy"
python altsteer/eval/eval_math_accuracy.py \
    --input "${ALTSTEER}" \
    --output "${EVALUATION_DIR}/math_accuracy_results.json"

echo "[6/9] Scoring refusal and constructive alternatives"
python altsteer/eval/eval_refusal_alternative_gpt4o.py \
    --altsteer "${ALTSTEER}" \
    --refusal_only "${REFUSAL_ONLY}" \
    --explanation_only "${EXPLANATION_ONLY}" \
    --output "${EVALUATION_DIR}/refusal_alternative_results.json"

echo "[7/9] Scoring compliance on XSTest safe prompts"
python altsteer/eval/eval_xstest_compliance_gpt4o.py \
    --input "${ALTSTEER}" \
    --output "${EVALUATION_DIR}/xstest_compliance_results.jsonl"

echo "[8/9] Scoring harmful leakage and the no-steering results"
for results in "${ALTSTEER}" "${REFUSAL_ONLY}" "${EXPLANATION_ONLY}"; do
    python altsteer/eval/eval_harmful_leakage_gpt4o.py --json_path "${results}" --resume
done
python altsteer/eval/eval_naive_all.py \
    --input "${NAIVE}" \
    --output_dir "${EVALUATION_DIR}/naive"

echo "[9/9] Running AlpacaEval"
if python -c "import alpaca_eval" 2>/dev/null; then
    ALPACA_ARGS=(
        --model_path "${MODEL_PATH}"
        --target_layer "${TARGET_LAYER}"
        --vector_dir "${VECTOR_DIR}"
        --refusal_strength "${REFUSAL_STRENGTH}"
        --explanation_strength "${EXPLANATION_STRENGTH}"
        --base_dir "${INFERENCE_DIR}"
    )
    python altsteer/src/generate_alpacaeval.py "${ALPACA_ARGS[@]}" --no_steering
    python altsteer/src/generate_alpacaeval.py "${ALPACA_ARGS[@]}"
    python altsteer/src/generate_alpacaeval.py "${ALPACA_ARGS[@]}" --run_evaluate
else
    echo "Skipping AlpacaEval because the alpaca-eval package is not installed."
fi

echo "Done. Results are in results/${MODEL_NAME}/"
