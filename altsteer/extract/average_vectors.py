"""Average the per-group refusal and explanation vectors into the final vectors."""

import argparse

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--refusal", required=True)
parser.add_argument("--explanation", required=True)
parser.add_argument("--refusal_output", required=True)
parser.add_argument("--explanation_output", required=True)
args = parser.parse_args()

refusal_vectors = torch.load(args.refusal)
refusal_vector = torch.stack(list(refusal_vectors.values())).mean(dim=0)
torch.save(refusal_vector, args.refusal_output)
print(f"Refusal vector {list(refusal_vector.shape)} saved to {args.refusal_output}")

explanation_vectors = torch.load(args.explanation)
explanation_vector = {
    layer: torch.stack(list(vectors.values())).mean(dim=0)
    for layer, vectors in explanation_vectors.items()
}
torch.save(explanation_vector, args.explanation_output)
print(f"Explanation vectors for {len(explanation_vector)} layers saved to {args.explanation_output}")
