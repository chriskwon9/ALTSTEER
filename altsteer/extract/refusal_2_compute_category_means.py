


import argparse
import torch
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()

# Load the hidden states
data = torch.load(args.input)

print(f"Total samples: {len(data)}")

# Group samples by category
category_samples = defaultdict(list)

for sample in data:
    category = sample['category']
    h_post_layers = sample['h_post_layers']  # Shape: [num_layers, hidden_size]
    category_samples[category].append(h_post_layers)

print(f"\nNumber of categories: {len(category_samples)}")
print("\nSamples per category:")
for category, samples in sorted(category_samples.items()):
    print(f"  {category}: {len(samples)} samples")

# Compute layer-wise mean for each category
category_mean_vectors = {}

for category, samples in category_samples.items():
    # Stack all samples: [num_samples, num_layers, hidden_size]
    stacked_samples = torch.stack(samples, dim=0)
    
    # Compute mean across samples (dim=0): [num_layers, hidden_size]
    mean_vector = stacked_samples.mean(dim=0)
    
    category_mean_vectors[category] = mean_vector
    
    print(f"\n{category}:")
    print(f"  Stacked shape: {stacked_samples.shape}")
    print(f"  Mean vector shape: {mean_vector.shape}")

# Save the results
output_path = args.output
torch.save(category_mean_vectors, output_path)

print(f"\n\nSaved category mean vectors to: {output_path}")
print(f"Total categories saved: {len(category_mean_vectors)}")
