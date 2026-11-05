"""
Math Accuracy Evaluation for GSM8K and MATH-500
Compares model responses against ground truth answers.

"""

import argparse
import json
import re
import os
from tqdm import tqdm


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCHMARK_DIR = os.path.join(REPO_ROOT, 'data', 'inputs')

# ────────────────────────────────────
# Answer extraction utilities
# ──────────────────────────────────────────────────────────────────────

def extract_boxed(text):
    """Extract content inside the last \\boxed{...} with proper brace matching."""
    idx = text.rfind('\\boxed{')
    if idx == -1:
        return None
    # Start after '\boxed{'
    start = idx + len('\\boxed{')
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    if depth == 0:
        return text[start:i-1].strip()
    return None


def extract_gsm8k_answer(solution_text):
    """Extract the final numeric answer from GSM8K solution (after ####)."""
    match = re.search(r'####\s*(.+)', solution_text)
    if match:
        return match.group(1).strip().replace(',', '')
    return None


def extract_math500_answer(answer_text):
    """Extract the answer from MATH-500 (already in 'answer' field).
    Remove \\boxed{} wrapper if present."""
    text = answer_text.strip()
    boxed = extract_boxed(text)
    if boxed:
        return boxed
    return text


def extract_answer_from_response(response_text):
    """Extract the final answer from a model's response.
    Tries multiple patterns commonly used by LLMs."""
    if not response_text:
        return None

    text = response_text.strip()

    # Pattern 1: \boxed{...} (with proper brace matching)
    boxed = extract_boxed(text)
    if boxed:
        return boxed

    # Pattern 2: "#### <answer>"
    match = re.search(r'####\s*(.+)', text)
    if match:
        return match.group(1).strip().replace(',', '')

    # Pattern 3: "The answer is <answer>" / "the answer is <answer>"
    match = re.search(r'[Tt]he\s+(?:final\s+)?answer\s+is[:\s]*(.+?)(?:\.|$)', text)
    if match:
        ans = match.group(1).strip().rstrip('.')
        return ans

    # Pattern 4: "= <number>" at the end
    match = re.search(r'=\s*([\d,]+\.?\d*)\s*$', text)
    if match:
        return match.group(1).replace(',', '')

    # Pattern 5: Last number in the response
    numbers = re.findall(r'[-+]?\d*\.?\d+', text)
    if numbers:
        return numbers[-1]

    return None


def normalize_answer(answer):
    """Normalize answer for comparison."""
    if answer is None:
        return None
    ans = str(answer).strip()
    # Remove $, \, spaces
    ans = ans.replace('$', '').replace('\\', '').replace(' ', '').replace(',', '')
    # Remove trailing period
    ans = ans.rstrip('.')
    # Try to convert to float for numeric comparison
    try:
        return str(float(ans))
    except ValueError:
        return ans.lower()


def answers_match(predicted, ground_truth):
    """Check if predicted answer matches ground truth."""
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)

    if pred_norm is None or gt_norm is None:
        return False

    # Direct string match
    if pred_norm == gt_norm:
        return True

    # Numeric comparison
    try:
        pred_float = float(pred_norm)
        gt_float = float(gt_norm)
        return abs(pred_float - gt_float) < 1e-6
    except (ValueError, TypeError):
        pass

    return False


# ──────────────────────────────────────────────────────────────────────
# Load ground truth solutions
# ──────────────────────────────────────────────────────────────────────

def load_ground_truth():
    """Load ground truth answers for GSM8K and MATH-500."""
    # GSM8K
    with open(os.path.join(BENCHMARK_DIR, 'gsm8k_with_answers.json')) as f:
        gsm8k_data = json.load(f)
    gsm8k_gt = {}
    for item in gsm8k_data:
        prompt = item['prompt']
        answer = extract_gsm8k_answer(item['solution'])
        gsm8k_gt[prompt] = answer

    # MATH-500
    with open(os.path.join(BENCHMARK_DIR, 'math500.jsonl')) as f:
        math500_data = [json.loads(line.strip()) for line in f]
    math500_gt = {}
    for item in math500_data:
        prompt = item['problem']
        answer = extract_math500_answer(item['answer'])
        math500_gt[prompt] = answer

    print(f"Ground truth loaded:")
    print(f"  GSM8K: {len(gsm8k_gt)} problems")
    print(f"  MATH-500: {len(math500_gt)} problems")

    return gsm8k_gt, math500_gt


# ──────────────────────────────────────────────────────────────────────
# Load model responses
# ──────────────────────────────────────────────────────────────────────

def load_responses(input_path):
    """Load model responses from an inference result file."""
    with open(input_path) as f:
        data = json.load(f)
    gsm8k = [(d['prompt'], d['response']) for d in data if d.get('dataset') == 'gsm8k']
    math500 = [(d['prompt'], d['response']) for d in data if d.get('dataset') == 'math_500']
    print(f"  gsm8k={len(gsm8k)}, math_500={len(math500)}")
    return {'gsm8k': gsm8k, 'math_500': math500}


# ──────────────────────────────────────────────────────────────────────
# Evaluate accuracy
# ──────────────────────────────────────────────────────────────────────

def evaluate_accuracy(responses, ground_truth, dataset_name):
    """Evaluate accuracy for a list of (prompt, response) pairs."""
    correct = 0
    total = 0
    errors = []

    for prompt, response in responses:
        gt_answer = ground_truth.get(prompt)
        if gt_answer is None:
            continue

        predicted = extract_answer_from_response(response)
        is_correct = answers_match(predicted, gt_answer)

        if is_correct:
            correct += 1
        else:
            errors.append({
                'prompt': prompt[:80],
                'predicted': predicted,
                'ground_truth': gt_answer,
            })
        total += 1

    accuracy = correct / total * 100 if total > 0 else 0
    return correct, total, accuracy, errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    print("=" * 60)
    print("Math Accuracy Evaluation (GSM8K & MATH-500)")
    print("=" * 60)

    print("\nLoading ground truth...")
    gsm8k_gt, math500_gt = load_ground_truth()

    print("\nLoading model responses...")
    responses = load_responses(args.input)

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)

    summary = {}
    for ds_name, gt in [('gsm8k', gsm8k_gt), ('math_500', math500_gt)]:
        if not responses[ds_name]:
            print(f"  {ds_name}: no data")
            continue
        correct, total, accuracy, errors = evaluate_accuracy(responses[ds_name], gt, ds_name)
        print(f"  {ds_name}: {correct}/{total} = {accuracy:.1f}%")
        summary[ds_name] = {'correct': correct, 'total': total, 'accuracy': accuracy}

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
