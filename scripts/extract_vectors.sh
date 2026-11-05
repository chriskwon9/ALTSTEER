#!/usr/bin/env bash
# Build the refusal and explanation vectors for one model.
#
# Usage:
#   bash scripts/extract_vectors.sh <model_name>
#
# Needs configs/<model_name>.sh and
# data/anchors/<model_name>/beavertails_train_metadata.jsonl.
# The final vectors are written to results/<model_name>/vector_extraction/vectors/.

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

METADATA="data/anchors/${MODEL_NAME}/beavertails_train_metadata.jsonl"
if [[ ! -f "${METADATA}" ]]; then
    echo "Error: ${METADATA} not found. See \"Adding a Model\" in readme.md." >&2
    exit 1
fi

OUTPUT_DIR="${OUTPUT_DIR:-results/${MODEL_NAME}/vector_extraction}"
SRC=altsteer/extract
HIDDEN_DIR="${OUTPUT_DIR}/hidden_states"
INTERMEDIATE_DIR="${OUTPUT_DIR}/intermediate"
VECTOR_DIR="${OUTPUT_DIR}/vectors"
mkdir -p "${HIDDEN_DIR}" "${INTERMEDIATE_DIR}" "${VECTOR_DIR}"

echo "[1/8] Extracting hidden states for all prompts"
python "${SRC}/refusal_0_extract_hidden_states.py" \
    --model_path "${MODEL_PATH}" \
    --metadata "${METADATA}" \
    --output "${HIDDEN_DIR}/refusal_hidden_states_all.pt"

echo "[2/8] Splitting hidden states into refused and complied"
python "${SRC}/refusal_1_split_hidden_states.py" \
    --metadata "${METADATA}" \
    --hidden_states "${HIDDEN_DIR}/refusal_hidden_states_all.pt" \
    --refused_output "${HIDDEN_DIR}/refusal_hidden_states_refused.pt" \
    --complied_output "${HIDDEN_DIR}/refusal_hidden_states_complied.pt"

echo "[3/8] Computing group means"
python "${SRC}/refusal_2_compute_category_means.py" \
    --input "${HIDDEN_DIR}/refusal_hidden_states_refused.pt" \
    --output "${INTERMEDIATE_DIR}/means_refused.pt"
python "${SRC}/refusal_2_compute_category_means.py" \
    --input "${HIDDEN_DIR}/refusal_hidden_states_complied.pt" \
    --output "${INTERMEDIATE_DIR}/means_complied.pt"

echo "[4/8] Computing refusal vectors"
python "${SRC}/refusal_3_compute_refusal_vectors.py" \
    --refused "${INTERMEDIATE_DIR}/means_refused.pt" \
    --complied "${INTERMEDIATE_DIR}/means_complied.pt" \
    --output "${INTERMEDIATE_DIR}/refusal_vectors.pt"

echo "[5/8] Extracting hidden states under the two system prompts"
python "${SRC}/explanation_1_extract_hidden_states.py" \
    --model_path "${MODEL_PATH}" \
    --metadata "${METADATA}" \
    --target_layer "${TARGET_LAYER}" \
    --output "${HIDDEN_DIR}/explanation_hidden_states.pt"

echo "[6/8] Computing explanation vectors"
python "${SRC}/explanation_2_compute_explanation_vectors.py" \
    --input "${HIDDEN_DIR}/explanation_hidden_states.pt" \
    --output "${INTERMEDIATE_DIR}/explanation_vectors.pt"

echo "[7/8] Orthogonalizing explanation vectors against refusal vectors"
python "${SRC}/explanation_3_orthogonalize.py" \
    --explanation "${INTERMEDIATE_DIR}/explanation_vectors.pt" \
    --refusal "${INTERMEDIATE_DIR}/refusal_vectors.pt" \
    --output "${INTERMEDIATE_DIR}/explanation_vectors_orthogonal.pt"

echo "[8/8] Averaging into the final vectors"
python "${SRC}/average_vectors.py" \
    --refusal "${INTERMEDIATE_DIR}/refusal_vectors.pt" \
    --explanation "${INTERMEDIATE_DIR}/explanation_vectors_orthogonal.pt" \
    --refusal_output "${VECTOR_DIR}/refusal_vector.pt" \
    --explanation_output "${VECTOR_DIR}/explanation_vector.pt"

echo "Done. Final vectors are in ${VECTOR_DIR}"
