"""Generate a model's responses to the BeaverTails train prompts, without steering.

The output is the input of altsteer/eval/label_refusal_gpt4o.py.
Resume-safe: skips the rows already saved in the output file.
"""

import argparse
import json
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', required=True)
parser.add_argument('--metadata', required=True, help="JSONL with idx, prompt and category")
parser.add_argument('--output', required=True)
args = parser.parse_args()

output_file = args.output
output_dir = os.path.dirname(output_file)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()

with open(args.metadata, 'r') as f:
    samples = [json.loads(line) for line in f if line.strip()]
print(f"Loaded {len(samples)} prompts")

processed_count = 0
results = []
if os.path.exists(output_file):
    with open(output_file, 'r') as f:
        for line in f:
            results.append(json.loads(line))
            processed_count += 1
    print(f"Loaded {processed_count} existing results")

for i in tqdm(range(processed_count, len(samples)), desc="Generating"):
    sample = samples[i]

    messages = [{"role": "user", "content": sample['prompt']}]
    chat_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode only the generated tokens
    generated_ids = outputs[0, inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    results.append({
        "idx": sample['idx'],
        "prompt": sample['prompt'],
        "category": sample['category'],
        "model_response": generated_text
    })

    # Save every 10 prompts
    if (i + 1) % 10 == 0:
        with open(output_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

if len(results) > 0:
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

print(f"Saved {len(results)} responses to {output_file}")
