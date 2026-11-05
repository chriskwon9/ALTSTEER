import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--refused', required=True)
parser.add_argument('--complied', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()

# Load the two mean vector files
refused_vectors = torch.load(args.refused)
complied_vectors = torch.load(args.complied)

print("Refused categories:", list(refused_vectors.keys()))
print("Complied categories:", list(complied_vectors.keys()))

# Find common categories
refused_cats = set(refused_vectors.keys())
complied_cats = set(complied_vectors.keys())
common_cats = refused_cats & complied_cats

print(f"\nCommon categories: {len(common_cats)}")
print(f"Only in refused: {refused_cats - complied_cats}")
print(f"Only in complied: {complied_cats - refused_cats}")

# Compute difference vectors (refused - complied)
category_steering_vectors = {}

for category in sorted(common_cats):
    refused_vec = refused_vectors[category]  # Shape: [num_layers, hidden_size]
    complied_vec = complied_vectors[category]  # Shape: [num_layers, hidden_size]
    
    # Compute difference: refused - complied
    steering_vec = refused_vec - complied_vec
    
    category_steering_vectors[category] = steering_vec
    
    print(f"\n{category}:")
    print(f"  Refused shape: {refused_vec.shape}")
    print(f"  Complied shape: {complied_vec.shape}")
    print(f"  Steering vector shape: {steering_vec.shape}")
    print(f"  Norm: {torch.norm(steering_vec).item():.4f}")

# Save the steering vectors
output_path = args.output
torch.save(category_steering_vectors, output_path)

print(f"\n\nSaved steering vectors to: {output_path}")
print(f"Total categories: {len(category_steering_vectors)}")
print(f"\nEach steering vector has shape [num_layers, hidden_size]:")
print(f"  - Each layer")
print(f"  - hidden_size dimensions per layer")
