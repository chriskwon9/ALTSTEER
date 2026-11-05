"""
Explanation vector computation from system prompts

Input: a hidden state file
- hidden states under the explained_refusal system prompt
- hidden states under the simple_refusal system prompt
- vectors are computed per category for every extracted layer
"""

import argparse
import torch
import numpy as np
from collections import defaultdict

# Configuration
parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()

DATA_PATH = args.input
OUTPUT_FILE = args.output


def create_explanation_vectors_per_category(data):
    """
    Compute explanation vectors per category
    
    For each category:
        explained_refusal mean - simple_refusal mean
    
    Returns:
        dict: {
            'layer_<n>': {category: vector, ...},
            'layer_14': {category: vector, ...},
            ...
        }
    """
    print()
    print("=" * 80)
    print("Creating Category-specific Explanation Vectors (All extracted layers)")
    print("=" * 80)
    
    # Organize by category and layer
    layer_vectors = {}
    
    for layer_num in LAYERS:
        print(f"\n>>> Processing Layer {layer_num}")
        print(f"{'Category':<50} {'Samples':>10} {'Norm':>10}")
        print("-" * 80)
        
        category_explained = defaultdict(list)
        category_simple = defaultdict(list)
        
        # Collect vectors by category
        for sample in data:
            category = sample['category']
            h_explained = sample['h_post_explained_refusal'][f'layer_{layer_num}']
            h_simple = sample['h_post_simple_refusal'][f'layer_{layer_num}']
            
            category_explained[category].append(h_explained)
            category_simple[category].append(h_simple)
        
        # Compute vectors for each category
        category_vecs = {}
        
        for category in sorted(category_explained.keys()):
            explained = category_explained[category]
            simple = category_simple[category]
            
            # Stack and compute means
            explained_stack = torch.stack(explained, dim=0)  # [N, hidden_size]
            simple_stack = torch.stack(simple, dim=0)
            
            explained_mean = explained_stack.mean(dim=0)  # [hidden_size]
            simple_mean = simple_stack.mean(dim=0)
            
            # Compute explanation vector
            vec = explained_mean - simple_mean
            vec_norm = torch.norm(vec).item()
            
            category_vecs[category] = vec
            
            print(f"{category:<50} {len(explained):>10} {vec_norm:>10.4f}")
        
        layer_vectors[f'layer_{layer_num}'] = category_vecs
        print(f"\n✓ Layer {layer_num}: Created vectors for {len(category_vecs)} categories")
    
    return layer_vectors

def main():
    print("=" * 80)
    print("Explanation vector computation from system prompts (all extracted layers)")
    print("=" * 80)
    print()
    
    # Load data
    print(f"Loading data from: {DATA_PATH}")
    data = torch.load(DATA_PATH)
    print(f"Total samples: {len(data)}")
    global LAYERS
    LAYERS = [int(key.split('_')[1]) for key in data[0]['h_post_explained_refusal']]
    print()
    
    # Count categories
    from collections import Counter
    categories = [sample['category'] for sample in data]
    category_counts = Counter(categories)
    
    print("Category distribution:")
    for cat, count in category_counts.most_common():
        print(f"  {cat}: {count}")
    print()
    
    # Create category-specific vectors for all layers
    layer_vectors = create_explanation_vectors_per_category(data)
    
    # Save vectors
    torch.save(layer_vectors, OUTPUT_FILE)
    print(f"\n✓ Saved explanation vectors to: {OUTPUT_FILE}")
    
    # Print statistics for each layer
    print()
    print("=" * 80)
    print("Vector Statistics by Layer")
    print("=" * 80)
    
    for layer_num in LAYERS:
        layer_key = f'layer_{layer_num}'
        category_vecs = layer_vectors[layer_key]
        
        norms = [torch.norm(vec).item() for vec in category_vecs.values()]
        
        print(f"\nLayer {layer_num}:")
        print(f"  Categories: {len(category_vecs)}")
        print(f"  Vector norms:")
        print(f"    Mean: {np.mean(norms):.6f} (± {np.std(norms):.6f})")
        print(f"    Min: {np.min(norms):.6f}")
        print(f"    Max: {np.max(norms):.6f}")
    
    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print()
    print("Notes:")
    print("  - Based on system prompts (explained vs simple refusal)")
    print("  - Every sample has both conditions for the same prompt")
    print("  - All extracted layers are processed")
    print("  - Original vector norms are kept (no normalization)")
    print()
    print("Done!")

if __name__ == "__main__":
    main()
