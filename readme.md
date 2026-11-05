<div align="center">
<h1> ALTSTEER: Selective Safety Steering for Moving Beyond Hard Refusals to Constructive Alternatives </h1>

Hoejoon Kwon, Byeonggeuk Lim, Kahyeon Kim, YoungBin Kim

Chung-Ang University

[![EMNLP 2026](https://img.shields.io/badge/EMNLP-2026-green.svg)](https://aclanthology.org/2026.emnlp-main.839/)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-red.svg)](https://pytorch.org/)

[Paper](https://aclanthology.org/2026.emnlp-main.839/)

</div>

---

ALTSTEER is an inference-time safety steering method. It shifts responses to harmful requests from short refusals toward constructive safe alternatives, within a single inference pass and without training.

It first measures refusal alignment: the cosine similarity between the prompt's last-token hidden states and the refusal vector, averaged over all layers. If the average is above 0, it adds two vectors at a target layer while the response is generated. At decoding step `t` of `T`:

- **Refusal vector**: its strength is `r * (1 - sqrt(t / T))`, which falls from `r` to 0.
- **Alternative vector**: its strength is `e + (1 - e) * sqrt(t / T)`, which moves from `e` to 1. With the included Llama setting of `e = 1.0`, it stays at 1.0. The code and file names call this the explanation vector.

This repository includes vectors and settings for Llama-3.1-8B-Instruct: layer 13, `r = 1.5` and `e = 1.0`. See [Adding a Model](#-adding-a-model) to use another model.

## 🔧 Requirements

* Python 3.12
* PyTorch 2.9.1
* Transformers 4.57.6

### Installation

```bash
git clone https://github.com/chriskwon9/ALTSTEER
cd ALTSTEER
pip install -e .
```

Install a PyTorch build that supports your GPU if the pinned version does not.

The model is downloaded from Hugging Face by default, which requires access to [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct). To use a local copy:

```bash
export MODEL_PATH=/path/to/model
```

## 🚀 Quick Start

### 1. Download the Benchmarks

```bash
bash scripts/download_data.sh
```

This saves the benchmarks to `data/inputs/` in the format the code expects. See [Data](#-data) for the sources. Vector extraction doesn't need this step.

### 2. Vector Extraction

The vectors used in our experiments are in `vectors/llama31-8b-instruct/`, so this step is optional.

| File | Contents |
|---|---|
| `refusal_vector.pt` | Tensor of shape `[num_layers, hidden_size]` |
| `explanation_vector.pt` | Dictionary from `layer_<n>` to a tensor of shape `[hidden_size]`, from the target layer to the last layer |

To extract them yourself:

```bash
bash scripts/extract_vectors.sh llama31-8b-instruct
```

### 3. Evaluation

One script generates responses on every benchmark and scores them:

```bash
export OPENAI_API_KEY=your-api-key
bash scripts/evaluate.sh llama31-8b-instruct
```

Results are saved to `results/llama31-8b-instruct/`. If the script stops, run it again to continue from the finished results.

| Step | Script | What it does |
|---|---|---|
| 1 | `altsteer/src/generate_altsteer.py` | AltSteer on all seven benchmarks |
| 2 | `altsteer/src/generate_refusal_only.py` | Refusal vector only, on the four safety benchmarks |
| 3 | `altsteer/src/generate_explanation_only.py` | Explanation vector only, on the four safety benchmarks |
| 4 | `altsteer/src/generate_naive.py` | No steering, on all seven benchmarks |
| 5 | `altsteer/eval/eval_math_accuracy.py` | Accuracy on GSM8K and MATH-500, with no API calls |
| 6 | `altsteer/eval/eval_refusal_alternative_gpt4o.py` | Refusal and constructive alternatives |
| 7 | `altsteer/eval/eval_xstest_compliance_gpt4o.py` | Compliance on XSTest safe prompts |
| 8 | `altsteer/eval/eval_harmful_leakage_gpt4o.py`, `eval_naive_all.py` | Harmful information in responses, and all scores for the no-steering results |
| 9 | `altsteer/src/generate_alpacaeval.py` | AlpacaEval, skipped unless you run `pip install alpaca-eval` |

## 🧩 Adding a Model

The model must be a causal language model whose layers are at `model.model.layers` and whose chat template supports a system message.

1. **Add a settings file.** Copy `configs/llama31-8b-instruct.sh` to `configs/<model_name>.sh`, then set the model path, target layer and strengths. The included values are for Llama and need tuning for other models.
2. **Add refusal labels.** Refusals differ across models, so you need labels from your own model. Generate its responses to the prompts in `data/anchors/llama31-8b-instruct/beavertails_train_metadata.jsonl`, then label them with `altsteer/eval/label_refusal_gpt4o.py`. Save the result as `data/anchors/<model_name>/beavertails_train_metadata.jsonl`, with `idx`, `prompt`, `category` and `explicit_refusal` on each line. `explicit_refusal` must be `Yes` or `No`.
3. **Extract the vectors.**

   ```bash
   bash scripts/extract_vectors.sh <model_name>
   ```

4. **Move the vectors into place.**

   ```bash
   mkdir -p vectors/<model_name>
   cp results/<model_name>/vector_extraction/vectors/*.pt vectors/<model_name>/
   ```

5. **Evaluate.**

   ```bash
   bash scripts/evaluate.sh <model_name>
   ```

## 📊 Data

This repository only includes the BeaverTails prompts and refusal labels used to extract the vectors, in `data/anchors/`. `scripts/download_data.sh` downloads the benchmarks from their sources:

| Benchmark | Source | Prompts used |
|---|---|---|
| AdvBench | [llm-attacks](https://github.com/llm-attacks/llm-attacks/blob/main/data/advbench/harmful_behaviors.csv) | 520 |
| JailbreakBench | [JBB-Behaviors](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) | 100 |
| MaliciousInstruct | [Jailbreak_LLM](https://github.com/Princeton-SysML/Jailbreak_LLM/blob/main/data/MaliciousInstruct.txt) | 100 |
| BeaverTails | [BeaverTails `330k_test`](https://huggingface.co/datasets/PKU-Alignment/BeaverTails) | 625 |
| XSTest | [xstest](https://github.com/paul-rottger/xstest) | 250 safe prompts |
| GSM8K | [grade-school-math](https://github.com/openai/grade-school-math) | 1,319 |
| MATH-500 | [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | 500 |
| AlpacaEval | [alpaca_eval](https://huggingface.co/datasets/tatsu-lab/alpaca_eval), downloaded when it runs | 805 |

Our BeaverTails test sample is identified by its row numbers in `data/beavertails_test_indices.json`.

## 📚 Citation

```bibtex
@inproceedings{kwon-etal-2026-altsteer,
    title = "{ALTSTEER}: Selective Safety Steering for Moving Beyond Hard Refusals to Constructive Alternatives",
    author = "Kwon, Hoejoon  and
      Lim, Byeonggeuk  and
      Kim, Kahyeon  and
      Kim, YoungBin",
    editor = "Lee, Hung-Yi  and
      Dev, Sunipa  and
      Li, Junyi Jessy  and
      Elliott, Desmond",
    booktitle = "Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.emnlp-main.839/",
    doi = "10.18653/2026.emnlp-main.839",
    pages = "19455--19473",
    ISBN = "979-8-89176-459-0"
}
```


