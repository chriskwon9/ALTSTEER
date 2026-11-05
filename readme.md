<div align="center">
<h1> ALTSTEER: Selective Safety Steering for Moving Beyond Hard Refusals to Constructive Alternatives </h1>

<!-- TODO: add the full paper title to the heading above and a venue badge below. -->

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-red.svg)](https://pytorch.org/)

**The official implementation of AltSteer**

<!-- TODO: add the paper link. -->

</div>

---

AltSteer is a conditional activation steering method. It first checks whether a prompt is harmful, using the cosine similarity between the prompt's last-token hidden states and the refusal vector, averaged over all layers. If the average is above 0, it adds two vectors at a target layer while the response is generated:

- **Refusal vector**: its strength falls toward 0.
- **Explanation vector**: its strength rises toward 1.

This repository includes vectors and settings for Llama-3.1-8B-Instruct. See [Adding a Model](#-adding-a-model) to use another model.

## 🔧 Requirements

* Python 3.12
* PyTorch 2.9.1
* Transformers 4.57.6

### Installation

```bash
git clone https://github.com/<your-username>/AltSteer
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

Small numerical differences across environments can change generated responses, so results may not match ours exactly.

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

Some of these prompts are harmful by design. Use them for safety research only.

## 📚 Citation

<!-- TODO: add the BibTeX entry. -->

## License

The code is released under the [Apache License 2.0](LICENSE). The benchmark data and the Llama model remain subject to their own licenses.
