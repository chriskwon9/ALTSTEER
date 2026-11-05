"""
Naive inference (no steering) for all datasets:
  advbench, jailbreakbench, malicious_instruct, beavertails,
  gsm8k, math_500, xstest_safe

Output: naive_inference_all_results.jsonl
Resume-safe: skips already processed (dataset, prompt) pairs.
"""

import argparse
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()

model_path = args.model_path
output_file = args.output
os.makedirs(os.path.dirname(output_file), exist_ok=True)
MAX_NEW_TOKENS = 512

DATA_PATHS = [
    (os.path.join(REPO_ROOT, 'data', 'inputs', 'advbench.json'), 'advbench'),
    (os.path.join(REPO_ROOT, 'data', 'inputs', 'jailbreakbench.jsonl'), 'jailbreakbench'),
    (os.path.join(REPO_ROOT, 'data', 'inputs', 'malicious_instruct.jsonl'), 'malicious_instruct'),
    (os.path.join(REPO_ROOT, 'data', 'inputs', 'beavertails_test.jsonl'), 'beavertails'),
    (os.path.join(REPO_ROOT, 'data', 'inputs', 'math500.jsonl'), 'math_500'),
    (os.path.join(REPO_ROOT, 'data', 'inputs', 'gsm8k.json'), 'gsm8k'),
    (os.path.join(REPO_ROOT, 'data', 'inputs', 'xstest.json'), 'xstest_safe'),
]

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()
print("✓ Model loaded\n")

print("Loading data...")
all_samples = []
for path, dataset_name in DATA_PATHS:
    with open(path, 'r') as f:
        if path.endswith('.json'):
            items = json.load(f)
        else:
            items = [json.loads(line) for line in f]

    if dataset_name == 'xstest_safe':
        items = [item for item in items if item.get('label') == 'safe']

    for item in items:
        prompt = item.get('prompt') or item.get('problem', '')
        all_samples.append({
            'dataset': dataset_name,
            'prompt': prompt,
        })
    print(f"  {dataset_name}: {len(items) if dataset_name != 'xstest_safe' else len([i for i in items])}")

print(f"Total {len(all_samples)} samples\n")

processed_keys = set()  # (dataset, prompt)
if os.path.exists(output_file):
    with open(output_file, 'r') as f:
        for line in f:
            r = json.loads(line)
            processed_keys.add((r['dataset'], r['prompt']))
    print(f"Existing results: {len(processed_keys)}, skipping\n")

remaining = [s for s in all_samples if (s['dataset'], s['prompt']) not in processed_keys]
print(f"Remaining samples: {len(remaining)}\n")

if not remaining:
    print("All samples are already processed.")
    exit(0)

newly_processed = 0

for i, sample in enumerate(tqdm(remaining, desc="Naive Inference")):
    prompt = sample['prompt']
    dataset_name = sample['dataset']

    messages = [{"role": "user", "content": prompt}]
    chat_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0, inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    result = {
        'dataset': dataset_name,
        'prompt': prompt,
        'response': response,
    }

    with open(output_file, 'a') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')

    newly_processed += 1

print(f"\n✓ Done: {newly_processed} newly processed")
print(f"Results saved to: {output_file}")

counts = {}
with open(output_file) as f:
    for line in f:
        ds = json.loads(line)['dataset']
        counts[ds] = counts.get(ds, 0) + 1
print("\nFinal result count by dataset:")
for ds, cnt in counts.items():
    print(f"  {ds}: {cnt}")
