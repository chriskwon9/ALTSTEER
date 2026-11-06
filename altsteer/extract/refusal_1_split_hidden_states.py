import argparse
import torch
import json

parser = argparse.ArgumentParser()
parser.add_argument('--metadata', required=True)
parser.add_argument('--hidden_states', required=True)
parser.add_argument('--refused_output', required=True)
parser.add_argument('--complied_output', required=True)
args = parser.parse_args()

# Load metadata to get explicit_refusal information
print("Loading metadata...")
metadata = []
with open(args.metadata, 'r') as f:
    for line in f:
        metadata.append(json.loads(line))
print(f"Loaded {len(metadata)} metadata entries")

# Load hidden states
print("\nLoading hidden states...")
hidden_states = torch.load(args.hidden_states)
print(f"Loaded {len(hidden_states)} hidden state samples")

# Verify lengths match
assert len(metadata) == len(hidden_states), f"Metadata ({len(metadata)}) and hidden states ({len(hidden_states)}) length mismatch!"

# Split by explicit_refusal status
refused_samples = []
complied_samples = []

print("\nSplitting samples by explicit_refusal status...")
for i, (meta, hidden) in enumerate(zip(metadata, hidden_states)):
    # Add the original index and metadata to the hidden state dictionary
    sample_with_index = {
        'original_index': i,
        'category': meta['category'],
        't_post_inst_position': hidden['t_post_inst_position'],
        'h_post_layers': hidden['h_post_layers']
    }
    
    if meta['explicit_refusal'] == 'Yes':
        refused_samples.append(sample_with_index)
    else:
        complied_samples.append(sample_with_index)

print(f"\nSplit complete:")
print(f"  Refused samples: {len(refused_samples)}")
print(f"  Complied samples: {len(complied_samples)}")

# Save split datasets
refused_file = args.refused_output
complied_file = args.complied_output

print(f"\nSaving refused samples to {refused_file}...")
torch.save(refused_samples, refused_file)

print(f"Saving complied samples to {complied_file}...")
torch.save(complied_samples, complied_file)

print("\n✓ Done!")
print(f"  Refused: {len(refused_samples)} samples with original_index and category")
print(f"  Complied: {len(complied_samples)} samples with original_index and category")
print(f"\nEach sample now contains:")
print(f"  - original_index: index in the full dataset")
print(f"  - category: safety category")
print(f"  - t_post_inst_position: post instruction token position")
print(f"  - h_post_layers: [num_layers, hidden_size] hidden states")
