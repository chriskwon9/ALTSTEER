import argparse
import json
import torch
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', required=True)
parser.add_argument('--metadata', required=True)
parser.add_argument('--target_layer', type=int, required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()

# Set random seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
print(f"Random seed set to {RANDOM_SEED}\n")

# Load model and tokenizer
model_path = args.model_path
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()
print(f"✓ Model loaded ({model.config.num_hidden_layers} layers)\n")

print("Loading data...")
with open(args.metadata, 'r') as f:
    data = [json.loads(line) for line in f if line.strip()]
data = [sample for sample in data if sample['explicit_refusal'] == 'Yes']
print(f"✓ Data loaded\n")

prompts = []
categories = []
for sample in data:
    prompts.append(sample['prompt'])
    categories.append(sample['category'])

print(f"Loaded {len(prompts)} prompts\n")

system_prompts = {
    "explained_refusal": "Refuse with 1 short sentence; then provide why this request is unsafe, and suggest 1 safe alternative",
    "simple_refusal": "Refuse with 1 short sentence; no reasons, no alternatives, no extra details."
}

# Output files
output_pt_file = args.output

# Layers to extract: from the target layer to the last layer
LAYERS_TO_EXTRACT = list(range(args.target_layer, model.config.num_hidden_layers))

print(f"Layers to extract: {LAYERS_TO_EXTRACT}")
print(f"Position: t_post_inst (right after the assistant header)\n")

all_hidden_states = []

# Process each sample
for idx, (prompt, category) in enumerate(tqdm(zip(prompts, categories), total=len(prompts), desc="Processing")):
    sample_data = {
        "idx": idx,
        "prompt": prompt,
        "category": category
    }
    
    # Process for each system prompt
    for sys_key, sys_prompt in system_prompts.items():
        # Apply chat template
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ]
        
        full_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        full_tokens = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        full_length = full_tokens["input_ids"].shape[1]
        
        # t_post_inst: last token of full prompt (with assistant header)
        t_post_inst_pos = full_length - 1
        
        # Get hidden states
        with torch.no_grad():
            outputs = model(
                **full_tokens,
                output_hidden_states=True,
                return_dict=True
            )
        
        # Extract hidden states at t_post_inst position
        # hidden_states[0] is embedding layer (skip)
        # hidden_states[1] is layer 0, hidden_states[2] is layer 1, and so on
        hidden_states = outputs.hidden_states
        
        h_post_layers = {}
        for layer_num in LAYERS_TO_EXTRACT:
            # layer_num -> hidden_states index layer_num + 1
            layer_idx = layer_num + 1
            h_t_post_inst = hidden_states[layer_idx][0, t_post_inst_pos, :].to(torch.float16).cpu()
            h_post_layers[f"layer_{layer_num}"] = h_t_post_inst
        
        # Store in sample data
        sample_data[f"h_post_{sys_key}"] = h_post_layers
        sample_data[f"t_post_inst_pos_{sys_key}"] = int(t_post_inst_pos)
    
    all_hidden_states.append(sample_data)
    
    # Periodic save to prevent memory overflow
    if (idx + 1) % 500 == 0:
        print(f"\nCheckpoint: {idx + 1} processed")
        torch.save(all_hidden_states, output_pt_file)
        torch.cuda.empty_cache()

# Final save
print(f"\nSaving final results...")
torch.save(all_hidden_states, output_pt_file)

print(f"\n{'='*80}")
print(f"✓ Done!")
print(f"{'='*80}")
print(f"Hidden states saved to: {output_pt_file}")
print(f"Total samples: {len(all_hidden_states)}")
print(f"\nSample structure:")
print(f"  - idx: sample index")
print(f"  - prompt: prompt text")
print(f"  - category: category")
print(f"  - h_post_explained_refusal: dict from layer name to tensor")
print(f"  - h_post_simple_refusal: dict from layer name to tensor")
print(f"  - t_post_inst_pos_explained_refusal: position index")
print(f"  - t_post_inst_pos_simple_refusal: position index")
