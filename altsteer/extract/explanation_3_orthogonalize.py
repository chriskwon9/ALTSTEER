import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--explanation', required=True)
parser.add_argument('--refusal', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()

print("=" * 80)
print("Orthogonalizing Explanation Vectors")
print("=" * 80)
print()

# Load explanation vectors
explanation_path = args.explanation
explanation_vectors_all_layers = torch.load(explanation_path)

print(f"Loaded explanation vectors from: {explanation_path}")
print(f"Layers: {list(explanation_vectors_all_layers.keys())}")
print(f"Categories per layer: {len(next(iter(explanation_vectors_all_layers.values())))}")
print()

# Load refusal steering vectors (all layers)
refusal_path = args.refusal
refusal_vectors_full = torch.load(refusal_path)

print(f"Loaded refusal vectors from: {refusal_path}")
print(f"Categories: {len(refusal_vectors_full)}")
print(f"Vector shape: {list(refusal_vectors_full.values())[0].shape}")
print()

# Process each layer
LAYERS = [int(key.split('_')[1]) for key in explanation_vectors_all_layers]
orthogonal_vectors_all_layers = {}

for layer_num in LAYERS:
    print("=" * 80)
    print(f"Processing Layer {layer_num}")
    print("=" * 80)
    print()
    
    # Get explanation vectors for this layer
    layer_key = f'layer_{layer_num}'
    explanation_vectors = explanation_vectors_all_layers[layer_key]
    
    # Extract this layer from refusal vectors
    refusal_vectors = {}
    for category, vec in refusal_vectors_full.items():
        refusal_vectors[category] = vec[layer_num]  # Extract layer
    
    # Find the groups in both files
    explanation_cats = set(explanation_vectors.keys())
    refusal_cats = set(refusal_vectors.keys())
    common_cats = explanation_cats & refusal_cats
    
    print(f"Common categories: {len(common_cats)}")
    if explanation_cats - refusal_cats:
        print(f"Only in explanation: {explanation_cats - refusal_cats}")
    if refusal_cats - explanation_cats:
        print(f"Only in refusal: {refusal_cats - explanation_cats}")
    print()
    
    # Orthogonalize: remove refusal component from explanation vectors
    orthogonal_explanation_vectors = {}
    
    for category in sorted(common_cats):
        explain_vec = explanation_vectors[category]  # [hidden_size]
        refusal_vec = refusal_vectors[category]  # [hidden_size]
        
        # Compute projection of explain_vec onto refusal_vec
        # projection = (explain · refusal) / (refusal · refusal)
        dot_product = torch.sum(explain_vec * refusal_vec)
        refusal_norm_sq = torch.sum(refusal_vec * refusal_vec)
        
        projection_scalar = dot_product / refusal_norm_sq
        
        # Compute orthogonal component: explain - projection * refusal
        orthogonal_vec = explain_vec - projection_scalar * refusal_vec
        
        orthogonal_explanation_vectors[category] = orthogonal_vec
        
        # Compute norms for comparison
        explain_norm = torch.norm(explain_vec).item()
        refusal_norm = torch.norm(refusal_vec).item()
        orthogonal_norm = torch.norm(orthogonal_vec).item()
        
        # Compute cosine similarity before orthogonalization
        cosine_sim = dot_product / (explain_norm * refusal_norm)
        
        # Verify orthogonality
        orthogonality_check = torch.sum(orthogonal_vec * refusal_vec).item()
        
        print(f"{category[:40]:40s}: cos_sim={cosine_sim.item():6.4f} | "
              f"proj={projection_scalar.item():6.4f} | ortho={orthogonal_norm:6.4f} | "
              f"check={orthogonality_check:8.6f}")
    
    orthogonal_vectors_all_layers[layer_key] = orthogonal_explanation_vectors
    print()

# Save orthogonalized vectors
output_path = args.output
torch.save(orthogonal_vectors_all_layers, output_path)

print("=" * 80)
print("Summary")
print("=" * 80)
print(f"✓ Saved orthogonal explanation vectors to:")
print(f"  {output_path}")
print()
print(f"Structure:")
print(f"  - Layers: {list(orthogonal_vectors_all_layers.keys())}")
print(f"  - Categories per layer: {len(next(iter(orthogonal_vectors_all_layers.values())))}")
print(f"  - Vector shape: [hidden_size]")
print()
print("✓ Refusal component has been removed from explanation vectors!")
print("✓ Now the explanation vectors are orthogonal to refusal vectors (FNTP).")
