"""
AlpacaEval 2.0: AltSteer versus the same model without steering.

This script is run by scripts/evaluate.sh.
"""

import argparse
import json
import os
import random
import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ============================================================
# Configuration
# ============================================================
RANDOM_SEED = 42
CLASSIFIER_THRESHOLD = 0.0

MAX_NEW_TOKENS = 512


# ============================================================
# Dynamic Steering Hook
# ============================================================
class PrefillRefusalDecodeExplanationHook:
    """Apply dynamic steering during generation with sqrt transitions."""

    def __init__(self, refusal_vector, explanation_vector, layer_idx,
                 refusal_strength_start, explanation_strength_start,
                 max_new_tokens):
        self.refusal_vector = refusal_vector.detach().cpu() if refusal_vector is not None else None
        self.explanation_vector = explanation_vector.detach().cpu() if explanation_vector is not None else None
        self.layer_idx = layer_idx
        self.refusal_strength_start = refusal_strength_start
        self.explanation_strength_start = explanation_strength_start
        self.max_new_tokens = max_new_tokens
        self.generated_token_count = 0
        self.hooks = []

    def _make_hook(self):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            device = hidden_states.device
            dtype = hidden_states.dtype
            seq_len = hidden_states.shape[1]

            # Generation only (seq_len == 1): dynamic refusal (↓) + explanation (↑)
            if seq_len == 1:
                sqrt_ratio = (self.generated_token_count / self.max_new_tokens) ** 0.5

                if self.refusal_vector is not None:
                    strength = self.refusal_strength_start * (1.0 - sqrt_ratio)
                    vec = self.refusal_vector[self.layer_idx].to(device, dtype=dtype)
                    hidden_states = hidden_states + strength * vec.unsqueeze(0).unsqueeze(0)

                if self.explanation_vector is not None:
                    strength = self.explanation_strength_start + (1.0 - self.explanation_strength_start) * sqrt_ratio
                    vec = self.explanation_vector[self.layer_idx].to(device, dtype=dtype)
                    hidden_states = hidden_states + strength * vec.unsqueeze(0).unsqueeze(0)

                self.generated_token_count += 1

            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            return hidden_states

        return hook_fn

    def register(self, model):
        layer = model.model.layers[self.layer_idx]
        self.hooks.append(layer.register_forward_hook(self._make_hook()))

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


# ============================================================
# Naive generation (no steering)
# ============================================================
def generate_naive(model, tokenizer, prompt, device):
    messages = [{"role": "user", "content": prompt}]
    chat_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id, do_sample=False)

    gen_ids = outputs[0, inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return response


def detect_harmfulness(model, tokenizer, prompt, mean_fntp, device):
    """
    Decide harmfulness from the prompt's last-token hidden states (all layers)
    using their average cosine similarity with the FNTP mean vector.
    """
    messages = [{"role": "user", "content": prompt}]
    chat_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states = outputs.hidden_states  # length num_layers+1

    cos_sims = []
    for layer_idx in range(mean_fntp.shape[0]):
        last_tok = hidden_states[layer_idx + 1][0, -1, :]  # [hidden_size]
        fntp_vec = mean_fntp[layer_idx].to(last_tok.device, dtype=last_tok.dtype)
        cos = F.cosine_similarity(
            last_tok.unsqueeze(0), fntp_vec.unsqueeze(0)).item()
        cos_sims.append(cos)

    avg_sim = sum(cos_sims) / len(cos_sims)
    return avg_sim > CLASSIFIER_THRESHOLD, avg_sim


# ============================================================
# OURS generation (classifier + conditional dynamic steering)
# ============================================================
def generate_steered(model, tokenizer, prompt,
                     refusal_vector, explanation_vector, mean_fntp, device):
    """
    1) Prefill: decide harmfulness with detect_harmfulness()
    2) Harmful: generate with steering
       Harmless: generate without steering
    """
    is_harmful, avg_cos_sim = detect_harmfulness(
        model, tokenizer, prompt, mean_fntp, device)

    messages = [{"role": "user", "content": prompt}]
    chat_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(device)

    hook = None
    if is_harmful:
        hook = PrefillRefusalDecodeExplanationHook(
            refusal_vector=refusal_vector,
            explanation_vector=explanation_vector,
            layer_idx=TARGET_LAYER,
            refusal_strength_start=REFUSAL_STRENGTH_START,
            explanation_strength_start=EXPLANATION_STRENGTH_START,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        hook.register(model)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False, pad_token_id=tokenizer.eos_token_id)

    if hook is not None:
        hook.remove()

    gen_ids = outputs[0, inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return response, is_harmful, avg_cos_sim


# ============================================================
# Main
# ============================================================
def main():
    global TARGET_LAYER, REFUSAL_STRENGTH_START, EXPLANATION_STRENGTH_START
    parser = argparse.ArgumentParser(
        description="AlpacaEval 2.0: AltSteer vs no steering")
    parser.add_argument('--no_steering', action='store_true',
                        help='Generate without steering (baseline)')
    parser.add_argument('--run_evaluate', action='store_true',
                        help='Run the alpaca_eval comparison (needs OPENAI_API_KEY)')
    parser.add_argument('--base_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--target_layer', type=int, required=True)
    parser.add_argument('--vector_dir', type=str, required=True)
    parser.add_argument('--refusal_strength', type=float, required=True)
    parser.add_argument('--explanation_strength', type=float, required=True)
    args = parser.parse_args()
    TARGET_LAYER = args.target_layer
    REFUSAL_STRENGTH_START = args.refusal_strength
    EXPLANATION_STRENGTH_START = args.explanation_strength
    os.makedirs(args.base_dir, exist_ok=True)

    # Seed
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    if args.no_steering:
        label = "No steering"
        generator_name = "naive"
        results_filename = "naive_alpacaeval_outputs.json"
    else:
        label = "AltSteer"
        generator_name = "altsteer"
        results_filename = "ours_alpacaeval_outputs.json"

    print("=" * 80)
    print(f"AlpacaEval 2.0 — {label}")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nLoading model... (device: {device})")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map='auto',
        output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()
    print(f"  Model: {args.model_path}")

    refusal_vector = None
    explanation_vector = None
    mean_fntp = None

    if not args.no_steering:
        print("\nLoading steering vectors...")

        # Averaged refusal vector — [num_layers, hidden_size]
        refusal_vector = torch.load(os.path.join(args.vector_dir, 'refusal_vector.pt'))
        print(f"  Averaged refusal vector: {refusal_vector.shape}")

        mean_fntp = refusal_vector.clone()
        print(f"  Classifier FNTP mean vector: {mean_fntp.shape} (threshold={CLASSIFIER_THRESHOLD})")

        # Explanation vector — target layer only, [num_layers, hidden_size]
        averaged_explanation_single = torch.load(os.path.join(args.vector_dir, 'explanation_vector.pt'))[f'layer_{TARGET_LAYER}']
        explanation_vector = torch.zeros(model.config.num_hidden_layers, model.config.hidden_size)
        explanation_vector[TARGET_LAYER] = averaged_explanation_single
        print(f"  Explanation vector: {explanation_vector.shape}")

        print(f"\n  ✓ AltSteer dynamic steering is set")
        print(f"    classifier       : cosine sim > {CLASSIFIER_THRESHOLD} → harmful → steering ON")
        print(f"    refusal strength : {REFUSAL_STRENGTH_START} (falls with sqrt during generation)")
        print(f"    explain strength : {EXPLANATION_STRENGTH_START} → 1.0 (rises with sqrt during generation)")
        print(f"    target layer     : {TARGET_LAYER}")
    else:
        print("  ✓ Steering disabled")

    print("✓ Model loaded\n")

    print("Loading AlpacaEval data...")
    from huggingface_hub import hf_hub_download
    ds_path = hf_hub_download('tatsu-lab/alpaca_eval', 'alpaca_eval_gpt4_baseline.json', repo_type='dataset')
    with open(ds_path, 'r') as f:
        ds = json.load(f)
    instructions = [item['instruction'] for item in ds]
    dataset_names = [item['dataset'] for item in ds]
    print(f"  {len(instructions)} instructions in total")
    for name, cnt in sorted(Counter(dataset_names).items(), key=lambda x: -x[1]):
        print(f"    {name}: {cnt}")
    print()

    results_path = f'{args.base_dir}/{results_filename}'
    existing_outputs = []
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            existing_outputs = json.load(f)
        print(f"  Loaded existing results: {len(existing_outputs)}/{len(instructions)}")

    start_idx = len(existing_outputs)
    if start_idx >= len(instructions):
        print("  All instructions are already processed.")
    else:
        remaining = instructions[start_idx:]
        print(f"  Instructions to generate: {len(remaining)} (from idx {start_idx})\n")

        desc = f"alpacaeval({'naive' if args.no_steering else 'ours'})"

        if args.no_steering:
            for i in tqdm(range(len(remaining)), desc=desc):
                idx = start_idx + i
                answer = generate_naive(model, tokenizer, remaining[i], device)
                existing_outputs.append({
                    'instruction': instructions[idx],
                    'output': answer,
                    'generator': generator_name,
                    'dataset': dataset_names[idx],
                })
                if (i + 1) % 50 == 0:
                    with open(results_path, 'w') as f:
                        json.dump(existing_outputs, f, ensure_ascii=False, indent=2)
        else:
            n_harmful = 0
            n_harmless = 0
            for i in tqdm(range(len(remaining)), desc=desc):
                idx = start_idx + i
                response, is_harmful, avg_cos_sim = generate_steered(
                    model, tokenizer, remaining[i],
                    refusal_vector, explanation_vector, mean_fntp, device)
                if is_harmful:
                    n_harmful += 1
                else:
                    n_harmless += 1
                existing_outputs.append({
                    'instruction': instructions[idx],
                    'output': response,
                    'generator': generator_name,
                    'dataset': dataset_names[idx],
                    'is_harmful': is_harmful,
                    'avg_cosine_sim': round(avg_cos_sim, 4),
                })
                if (i + 1) % 50 == 0:
                    with open(results_path, 'w') as f:
                        json.dump(existing_outputs, f, ensure_ascii=False, indent=2)
            print(f"\n  Classifier results: harmful={n_harmful}, harmless={n_harmless}")

    with open(results_path, 'w') as f:
        json.dump(existing_outputs, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Results saved to: {results_path} ({len(existing_outputs)} items)")

    naive_path = f'{args.base_dir}/naive_alpacaeval_outputs.json'
    ours_path = f'{args.base_dir}/ours_alpacaeval_outputs.json'
    both_exist = os.path.exists(naive_path) and os.path.exists(ours_path)

    if args.run_evaluate and both_exist:
        print("\n--- AltSteer vs no steering win rate ---")
        print("  ⚠️  OPENAI_API_KEY is required\n")
        output_dir = f'{args.base_dir}/alpacaeval_results'
        os.makedirs(output_dir, exist_ok=True)
        try:
            from alpaca_eval import evaluate
            results = evaluate(
                model_outputs=ours_path,
                reference_outputs=naive_path,
                name='altsteer',
                output_path=output_dir,
                is_return_instead_of_print=True,
            )
            print("\n--- OURS vs Naive Win Rate ---")
            print(results)
        except Exception as e:
            print(f"\n  ⚠️  alpaca_eval evaluate failed: {e}")
            print(f"  Run manually:")
            print(f"    alpaca_eval --model_outputs {ours_path} \\")
            print(f"                --reference_outputs {naive_path}")

    elif args.run_evaluate:
        output_dir = f'{args.base_dir}/alpacaeval_results'
        os.makedirs(output_dir, exist_ok=True)
        try:
            from alpaca_eval import evaluate
            results = evaluate(
                model_outputs=results_path,
                name=generator_name,
                output_path=output_dir,
                is_return_instead_of_print=True,
            )
            print(results)
        except Exception as e:
            print(f"\n  ⚠️  alpaca_eval evaluate failed: {e}")
    else:
        print(f"\n💡 Next steps:")
        if both_exist:
            print(f"  [AltSteer vs no steering]")
            print(f"    python altsteer/src/generate_alpacaeval.py --run_evaluate")
            print(f"  Or:")
            print(f"    alpaca_eval --model_outputs {ours_path} \\")
            print(f"                --reference_outputs {naive_path}")
        else:
            missing = 'naive' if not os.path.exists(naive_path) else 'ours'
            print(f"  ⚠️  {missing} results not found. Generate them first:")
            if missing == 'naive':
                print(f"    python altsteer/src/generate_alpacaeval.py --no_steering")
            else:
                print(f"    python altsteer/src/generate_alpacaeval.py")

    print("\n✓ Done")


if __name__ == "__main__":
    main()
