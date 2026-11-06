"""Explanation-Only Steering Script with Harmfulness Detection (No Refusal)
Prefill: Detect harmfulness via cosine similarity (all layers avg) with FNTP mean vector
  - If avg cosine sim > 0: Apply explanation steering during generation (NO refusal)
  - If avg cosine sim <= 0: No steering (naive model inference)

Steering (generation only, target layer):
    Explanation: α_exp,t = EXPLANATION_STRENGTH_START × (1 + sqrt(t / max_tokens))
    Refusal: DISABLED

Datasets: advbench, jailbreakbench, malicious_instruct, beavertails
"""

import argparse
import json
import os
import torch
import torch.nn.functional as F
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', required=True)
parser.add_argument('--target_layer', type=int, required=True)
parser.add_argument('--vector_dir', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--explanation_strength', type=float, required=True)
args = parser.parse_args()

# Configuration
RANDOM_SEED = 42
TARGET_LAYER = args.target_layer
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model_path = args.model_path
data_paths = [
    (os.path.join(REPO_ROOT, 'data', 'inputs', 'advbench.json'), 'advbench'),
    (os.path.join(REPO_ROOT, 'data', 'inputs', 'jailbreakbench.jsonl'), 'jailbreakbench'),
    (os.path.join(REPO_ROOT, 'data', 'inputs', 'malicious_instruct.jsonl'), 'malicious_instruct'),
    (os.path.join(REPO_ROOT, 'data', 'inputs', 'beavertails_test.jsonl'), 'beavertails'),
]

# Steering vectors
refusal_vector_path = os.path.join(args.vector_dir, 'refusal_vector.pt')
explanation_vector_path = os.path.join(args.vector_dir, 'explanation_vector.pt')

# Output path
output_path = args.output
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Steering strength parameters
EXPLANATION_STRENGTH_START = args.explanation_strength

# Set random seeds
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load model and tokenizer
print("\nLoading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map='auto',
    output_hidden_states=True
)
model.eval()

# Load steering vectors (FNTP for detection only)
print("\nLoading FNFP refusal steering vectors (for detection only)...")
averaged_refusal_vector = torch.load(refusal_vector_path)
print(f"Averaged refusal vector shape: {averaged_refusal_vector.shape}")

# FNTP mean vector for detection
mean_fntp = averaged_refusal_vector
print(f"FNTP mean vector for detection: {mean_fntp.shape}")

print("\nLoading explanation steering vectors...")
averaged_explanation_vector_single = torch.load(explanation_vector_path)[f'layer_{TARGET_LAYER}']
print(f"Averaged explanation vector shape: {averaged_explanation_vector_single.shape}")

# Create full tensor for explanation vector
averaged_explanation_vector = torch.zeros(model.config.num_hidden_layers, model.config.hidden_size)
averaged_explanation_vector[TARGET_LAYER] = averaged_explanation_vector_single
print(f"Full explanation vector shape: {averaged_explanation_vector.shape}")

# Load data from multiple datasets
print("\nLoading datasets...")
data = []
for data_path, dataset_name in data_paths:
    with open(data_path, 'r') as f:
        if data_path.endswith('.json'):
            items = json.load(f)
        else:
            items = [json.loads(line.strip()) for line in f]
    for item in items:
        if 'prompt' not in item and 'problem' in item:
            item['prompt'] = item['problem']
        item['dataset'] = dataset_name
    data.extend(items)
    print(f"  {dataset_name}: {len(items)} samples")
print(f"Total loaded: {len(data)} samples")

# Load existing results if available (resume from checkpoint)
existing_results = []
processed_indices = set()
if os.path.exists(output_path):
    print(f"\n{'='*60}")
    print(f"RESUME MODE: Loading existing results from {output_path}")
    print(f"{'='*60}")
    try:
        with open(output_path, 'r') as f:
            existing_results = json.load(f)
        
        if not isinstance(existing_results, list):
            print(f"Warning: Invalid format in {output_path}, starting fresh")
            existing_results = []
        else:
            for result in existing_results:
                if 'index' in result:
                    processed_indices.add(result['index'])
            
            print(f"✓ Loaded {len(existing_results)} existing results")
            print(f"✓ Will skip {len(processed_indices)} already-processed indices")
            print(f"✓ Will process {len(data) - len(processed_indices)} remaining samples")
    except json.JSONDecodeError as e:
        print(f"Warning: Could not parse {output_path} ({e}), starting fresh")
        existing_results = []
    except Exception as e:
        print(f"Warning: Error loading {output_path} ({e}), starting fresh")
        existing_results = []
else:
    print(f"\n{'='*60}")
    print(f"NEW RUN: No existing results found, starting from scratch")
    print(f"{'='*60}")

# Explanation-Only Steering Hook
class ExplanationOnlyHook:
    def __init__(self, explanation_vector=None, layer_idx=None,
                 max_new_tokens=512, prefill_length=0):
        """
        Apply explanation-only steering during generation with square root transition.
        NO refusal steering applied.
        
        explanation_vector: [num_layers, hidden_size] tensor for explanation steering
        layer_idx: specific layer to apply steering
        max_new_tokens: maximum number of tokens to generate
        prefill_length: length of prefill tokens
        """
        self.explanation_vector = explanation_vector.detach().cpu() if explanation_vector is not None else None
        self.layer_idx = layer_idx
        self.max_new_tokens = max_new_tokens
        self.prefill_length = prefill_length
        self.generated_token_count = 0
        self.hooks = []
    
    def make_steering_hook(self):
        """Create a hook function for the target layer"""
        def hook_fn(module, input, output):
            """Hook function to modify residual stream"""
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            device = hidden_states.device
            dtype = hidden_states.dtype
            seq_len = hidden_states.shape[1]
            
            # Prefill phase: no steering
            if seq_len > 1:
                pass
            
            # Apply explanation-only steering during generation
            if seq_len == 1:
                sqrt_ratio = (self.generated_token_count / self.max_new_tokens) ** 0.5
                
                # Apply explanation (increasing with sqrt, starting at EXPLANATION_STRENGTH_START)
                if self.explanation_vector is not None:
                    explanation_strength_current = EXPLANATION_STRENGTH_START * (1.0 + sqrt_ratio)
                    explain_vec = self.explanation_vector[self.layer_idx].to(device, dtype=dtype)
                    hidden_states = hidden_states + explanation_strength_current * explain_vec.unsqueeze(0).unsqueeze(0)
                
                self.generated_token_count += 1
            
            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            else:
                return hidden_states
        
        return hook_fn
    
    def register(self, model):
        """Register hook to specific layer's residual stream"""
        layer = model.model.layers[self.layer_idx]
        hook = layer.register_forward_hook(self.make_steering_hook())
        self.hooks.append(hook)
    
    def remove(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

def detect_harmfulness(prompt):
    """
    Detect harmfulness by computing cosine similarity between
    the last token hidden states (all layers) and the FNTP mean vector.
    """
    messages = [{"role": "user", "content": prompt}]
    chat_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    hidden_states = outputs.hidden_states
    
    cos_sims = []
    for layer_idx in range(model.config.num_hidden_layers):
        hs = hidden_states[layer_idx + 1]
        last_tok = hs[0, -1, :]
        fntp_vec = mean_fntp[layer_idx].to(last_tok.device).to(last_tok.dtype)
        cos = F.cosine_similarity(last_tok.unsqueeze(0), fntp_vec.unsqueeze(0)).item()
        cos_sims.append(cos)
    
    avg_sim = sum(cos_sims) / len(cos_sims)
    return avg_sim > 0, avg_sim, cos_sims


def generate_response(prompt, max_new_tokens=512):
    """
    Generate response with explanation-only steering during generation.
    No refusal steering applied.
    """
    # Step 1: Detect harmfulness via cosine similarity
    is_harmful, avg_cosine_sim, per_layer_sims = detect_harmfulness(prompt)
    
    # Apply chat template
    messages = [{"role": "user", "content": prompt}]
    chat_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(device)
    prefill_length = inputs["input_ids"].shape[1]
    
    # Step 2: Conditionally apply explanation-only steering based on detection
    if is_harmful:
        hook_manager = ExplanationOnlyHook(
            explanation_vector=averaged_explanation_vector,
            layer_idx=TARGET_LAYER,
            max_new_tokens=max_new_tokens,
            prefill_length=prefill_length
        )
        hook_manager.register(model)
    else:
        hook_manager = None
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Get strengths used during generation
    explanation_strengths = []
    if hook_manager is not None:
        num_generated = hook_manager.generated_token_count
        for t in range(num_generated):
            sqrt_ratio = (t / max_new_tokens) ** 0.5
            explanation_strengths.append(EXPLANATION_STRENGTH_START * (1.0 + sqrt_ratio))
        hook_manager.remove()
    
    # Decode only generated tokens
    gen_ids = outputs[0, inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(gen_ids, skip_special_tokens=True)
    
    return response, explanation_strengths, is_harmful, avg_cosine_sim

# Process samples
print("\nGenerating responses with detection + explanation-only steering (NO refusal)...")
print(f"Target layer: {TARGET_LAYER}")
print(f"Detection: avg cosine sim (all layers) with FNTP mean vector")
print(f"  > 0: Apply explanation-only steering during generation")
print(f"  <= 0: No steering (naive inference)")
print(f"Explanation (generation): {EXPLANATION_STRENGTH_START} × (1 + sqrt(t/max_tokens))")
print(f"Refusal: DISABLED")

save_interval = 10

# Calculate remaining samples
remaining_samples = [i for i, sample in enumerate(data) if i not in processed_indices]
print(f"\n{'='*60}")
print(f"PROCESSING SUMMARY:")
print(f"  Total samples in dataset: {len(data)}")
print(f"  Already processed: {len(processed_indices)} samples")
print(f"  Remaining to process: {len(remaining_samples)} samples")
if remaining_samples:
    print(f"  First remaining index: {remaining_samples[0]}")
    print(f"  Last remaining index: {remaining_samples[-1]}")
print(f"{'='*60}")

results = existing_results.copy()
newly_processed = 0

# Create iterator for remaining samples only
remaining_data = [(i, data[i]) for i in remaining_samples]

for idx, (i, sample) in enumerate(tqdm(remaining_data, desc="Processing remaining samples", total=len(remaining_samples))):
    prompt = sample['prompt']
    dataset_name = sample.get('dataset', 'unknown')
    print(f"\n[Index {i} | Progress {idx+1}/{len(remaining_samples)}] Dataset: {dataset_name}")
    
    try:
        response, explanation_strengths, is_harmful, avg_cosine_sim = generate_response(prompt)
        
        steering_applied = is_harmful
        print(f"  Detection: avg_cosine_sim={avg_cosine_sim:.4f}, harmful={is_harmful}, steering={'EXPLAIN-ONLY' if steering_applied else 'OFF'}")
        
        result = {
            'index': i,
            'prompt': prompt,
            'dataset': dataset_name,
            'is_harmful': is_harmful,
            'avg_cosine_sim': avg_cosine_sim,
            'steering_applied': steering_applied,
            'refusal_strength_range': 'DISABLED',
            'explanation_strength_range': f"{EXPLANATION_STRENGTH_START} → {2 * EXPLANATION_STRENGTH_START} (sqrt)" if steering_applied else "N/A (no steering)",
            'vector_type': 'averaged_explain_only' if steering_applied else 'none',
            'num_generated_tokens': len(explanation_strengths) if explanation_strengths else len(response.split()),
            'refusal_strengths': [],
            'explanation_strengths': (explanation_strengths[:10] if len(explanation_strengths) > 10 else explanation_strengths) if explanation_strengths else [],
            'response': response,
            'response_length': len(response)
        }
        
        print(f"  Generated tokens: {result['num_generated_tokens']}")
        if explanation_strengths:
            print(f"  Explanation: {explanation_strengths[0]:.4f} → {explanation_strengths[-1]:.4f}")
        else:
            print(f"  No steering applied (naive inference)")
        print(f"  Response length: {len(response)} chars")
        
        results.append(result)
        newly_processed += 1
        
        if newly_processed % save_interval == 0:
            print(f"\n{'='*50}")
            print(f"💾 CHECKPOINT: Saving progress...")
            print(f"   Newly processed: {newly_processed}")
            print(f"   Total in file: {len(results)}")
            print(f"   Remaining: {len(remaining_samples) - newly_processed}")
            print(f"{'='*50}")
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"✓ Checkpoint saved to {output_path}")
        
    except Exception as e:
        print(f"\n❌ Error processing sample {i}: {e}")
        continue

# Final save
print(f"\n{'='*60}")
print(f"FINAL SAVE: Saving all results to {output_path}")
print(f"{'='*60}")
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print(f"✓ COMPLETED SUCCESSFULLY!")
print(f"{'='*60}")
print(f"Session summary:")
print(f"  • Started with: {len(processed_indices)} already-processed samples")
print(f"  • Newly processed: {newly_processed} samples")
print(f"  • Total in output file: {len(results)} samples")
print(f"  • Output file: {output_path}")
print(f"\nSteering configuration:")
print(f"  Detection: avg cosine sim (all 32 layers) > 0 → steering ON")
print(f"  Refusal: DISABLED")
print(f"  Explanation (generation): {EXPLANATION_STRENGTH_START} × (1 + sqrt(t/max_tokens))")
print(f"  Steering layer: {TARGET_LAYER}")
