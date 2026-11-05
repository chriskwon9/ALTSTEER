"""Download the benchmarks from their public sources into data/inputs/."""

import argparse
import csv
import io
import json
import lzma
import os
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ADVBENCH_URL = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
JAILBREAKBENCH_URL = "https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/resolve/main/data/harmful-behaviors.csv"
MALICIOUS_INSTRUCT_URL = "https://raw.githubusercontent.com/Princeton-SysML/Jailbreak_LLM/main/data/MaliciousInstruct.txt"
BEAVERTAILS_URL = "https://huggingface.co/datasets/PKU-Alignment/BeaverTails/resolve/main/round0/330k/test.jsonl.xz"
XSTEST_URL = "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv"
GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
MATH500_URL = "https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl"


def download(url):
    print(f"Downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def read_csv(url):
    return list(csv.DictReader(io.StringIO(download(url).decode("utf-8"))))


def read_jsonl(content):
    return [json.loads(line) for line in content.decode("utf-8").splitlines() if line.strip()]


def write_json(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def check_count(name, rows, expected):
    if len(rows) != expected:
        raise ValueError(f"{name}: expected {expected} rows, got {len(rows)}. The source may have changed.")
    print(f"  {name}: {len(rows)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=os.path.join(REPO_ROOT, "data", "inputs"))
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    path = lambda name: os.path.join(args.output_dir, name)

    rows = [{"prompt": row["goal"]} for row in read_csv(ADVBENCH_URL)]
    check_count("advbench", rows, 520)
    write_json(path("advbench.json"), rows)

    rows = [{"Index": i, "prompt": row["Goal"]} for i, row in enumerate(read_csv(JAILBREAKBENCH_URL))]
    check_count("jailbreakbench", rows, 100)
    write_jsonl(path("jailbreakbench.jsonl"), rows)

    lines = [line.strip() for line in download(MALICIOUS_INSTRUCT_URL).decode("utf-8").splitlines() if line.strip()]
    rows = [{"Index": i, "prompt": line} for i, line in enumerate(lines)]
    check_count("malicious_instruct", rows, 100)
    write_jsonl(path("malicious_instruct.jsonl"), rows)

    # data/beavertails_test_indices.json holds the row numbers of our test sample in 330k_test.
    source = read_jsonl(lzma.decompress(download(BEAVERTAILS_URL)))
    with open(os.path.join(REPO_ROOT, "data", "beavertails_test_indices.json"), "r") as f:
        indices = json.load(f)
    rows = []
    for index in indices:
        row = source[index]
        category = next(name for name, is_true in row["category"].items() if is_true)
        rows.append({"prompt": row["prompt"], "category": category})
    check_count("beavertails", rows, 625)
    write_jsonl(path("beavertails_test.jsonl"), rows)

    rows = [{"prompt": row["prompt"], "label": row["label"]} for row in read_csv(XSTEST_URL)]
    check_count("xstest", rows, 450)
    write_json(path("xstest.json"), rows)

    source = read_jsonl(download(GSM8K_URL))
    check_count("gsm8k", source, 1319)
    write_json(path("gsm8k.json"), [{"prompt": row["question"]} for row in source])
    write_json(
        path("gsm8k_with_answers.json"),
        [{"prompt": row["question"], "solution": row["answer"]} for row in source],
    )

    rows = read_jsonl(download(MATH500_URL))
    check_count("math500", rows, 500)
    write_jsonl(path("math500.jsonl"), rows)

    print(f"Saved the benchmarks to {args.output_dir}")


if __name__ == "__main__":
    main()
