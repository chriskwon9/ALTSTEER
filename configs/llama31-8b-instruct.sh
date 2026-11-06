# Settings for Llama-3.1-8B-Instruct.
MODEL_PATH="${MODEL_PATH:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
TARGET_LAYER=13

# AltSteer strengths at decoding step t of T (the maximum new tokens):
#   refusal     = REFUSAL_STRENGTH * (1 - sqrt(t / T))
#   explanation = EXPLANATION_STRENGTH * (1 + sqrt(t / T))
REFUSAL_STRENGTH=1.5
EXPLANATION_STRENGTH=1.0

# Single-vector variants
REFUSAL_ONLY_STRENGTH=1.0
EXPLANATION_ONLY_STRENGTH=1.0
