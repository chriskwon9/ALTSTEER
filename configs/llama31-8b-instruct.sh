# Settings for Llama-3.1-8B-Instruct.
MODEL_PATH="${MODEL_PATH:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
TARGET_LAYER=13

# AltSteer: refusal falls from REFUSAL_STRENGTH toward 0,
# and explanation rises from EXPLANATION_STRENGTH toward 1.
REFUSAL_STRENGTH=1.0
EXPLANATION_STRENGTH=0.7

# Single-vector variants
REFUSAL_ONLY_STRENGTH=1.0
EXPLANATION_ONLY_STRENGTH=1.0
