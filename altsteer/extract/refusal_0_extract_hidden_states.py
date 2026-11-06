"""Extract last-token hidden states of every layer for the BeaverTails train prompts.

Each sample stores h_post_layers with shape [num_layers, hidden_size], where
index i is the output of model.model.layers[i] (hidden_states[i + 1]).
"""

import argparse
import json

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", required=True)
parser.add_argument("--metadata", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_path)
model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()

with open(args.metadata, "r") as f:
    metadata = [json.loads(line) for line in f if line.strip()]
print(f"Loaded {len(metadata)} prompts")

# Stop here if the labels are incomplete: every label other than "Yes" would count as complied.
invalid_rows = [
    i for i, sample in enumerate(metadata)
    if "category" not in sample or sample.get("explicit_refusal") not in ("Yes", "No")
]
if invalid_rows:
    raise SystemExit(
        f"Error: {len(invalid_rows)} rows in {args.metadata} need a category and an "
        f"explicit_refusal of Yes or No. First invalid row: {invalid_rows[0]}."
    )

all_hidden_states = []
for sample in tqdm(metadata, desc="Extracting hidden states"):
    messages = [{"role": "user", "content": sample["prompt"]}]
    full_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_tokens = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    t_post_inst_pos = full_tokens["input_ids"].shape[1] - 1

    with torch.no_grad():
        outputs = model(**full_tokens, output_hidden_states=True, return_dict=True)

    # hidden_states[0] is the embedding output, so layer i is hidden_states[i + 1].
    h_post_layers = torch.stack(
        [
            layer_output[0, t_post_inst_pos, :].to(torch.float16).cpu()
            for layer_output in outputs.hidden_states[1:]
        ],
        dim=0,
    )
    all_hidden_states.append(
        {
            "t_post_inst_position": int(t_post_inst_pos),
            "h_post_layers": h_post_layers,
        }
    )

torch.save(all_hidden_states, args.output)
print(f"Saved {len(all_hidden_states)} samples to {args.output}")
