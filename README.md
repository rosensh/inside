# INSIDE: LLM Student Simulators That Reason Behind Their Actions

Rose Niousha, Minwoo Kang, Narges Norouzi  
Department of Electrical Engineering and Computer Sciences  
University of California, Berkeley  
`{rose.n, minwoo_kang, norouzi}@berkeley.edu`  
(Accepted at COLM 2026)

<p align="center">
  <img src="logo/inside-logo.jpg" alt="INSIDE logo" width="240">
</p>

**TL;DR.** LLM-based simulators can reproduce observable student actions while missing the reasoning behind them. INSIDE (Internal Student Dialogue) fine-tunes LLMs not only to act like students, but also to think like them, by generating internal dialogue grounded in Bloom's Taxonomy across cognitive, affective, and action dimensions before producing the next student code submission. INSIDE improves simulation fidelity by better matching real student code generation and improves reasoning alignment, reaching up to 57.9% alignment compared to prompting baselines.

If you are working on student simulation for intelligent tutoring systems, AI tutor evaluation, or user simulation more broadly, this repo provides the pipeline we used once student trajectory data is available: generate internal dialogue traces, fine-tune student simulators, generate synthetic student submissions, and evaluate action fidelity and reasoning alignment.


## Start Here

The intended workflow is:

1. Prepare your student trajectory data in the format shown in `data_examples/`.
2. Generate internal dialogue traces with `internal_dialogue/`.
3. Fine-tune student simulators with `fine_tuning/`.
4. Generate model outputs with `data_generation/`.
5. Run action-fidelity and think-alignment evaluations with `eval/`.

## Layout

```text
data_examples/            Synthetic data-format examples
internal_dialogue/        Retrospective think-trace generation
fine_tuning/              LoRA/SFT training
data_generation/          Local, API, and OpenAI-compatible generation
eval/fidelity_eval/       Code formatting, static metrics, optional doctest grading
eval/think_eval/          LLM-judge think-alignment evaluation
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For API calls, set:

```bash
export OPENAI_API_KEY="..."
# Optional: any OpenAI-compatible endpoint.
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

The paper used GPT-family models for teacher trace generation and LLM judging. You can use OpenAI models, an OpenAI-compatible endpoint, or another provider of your choice by adapting the API client/config.

## 1. Prepare Data

Start from the synthetic examples:

- `data_examples/train_exp1_synthetic.jsonl`: code-only target for Experiment 1
- `data_examples/train_exp2_synthetic.jsonl`: `<think>...</think>` plus code target for INSIDE
- `data_examples/generation_synthetic.jsonl`: example model output before evaluation formatting

Each training row has:

- `INPUT`: problem text, skeleton code, prior submissions, prior tutor feedback, and metadata
- `OUTPUT`: the next student submission

For INSIDE training, `OUTPUT` starts with a lowercase think trace:

```text
<think>
...
</think>

student code here
```

## 2. Generate Internal Dialogue

If your data has observed next submissions but no think traces, generate retrospective traces:

```bash
python internal_dialogue/main.py \
  --config internal_dialogue/config.example.json \
  --input data_examples/train_exp1_synthetic.jsonl \
  --output outputs/train_exp2_traced.jsonl \
  --max_workers 4
```

Use the resulting traced JSONL as the `exp2` training file.

## 3. Fine-Tuning

Paper experiment names:

- `exp1`: generate the next code submission directly
- `exp2`: INSIDE, generate internal dialogue followed by code

Example INSIDE fine-tuning command:

```bash
python fine_tuning/fine_tune.py \
  --experiment_name exp2 \
  --dataset_file data_examples/train_exp2_synthetic.jsonl \
  --model_name Qwen/Qwen2.5-Coder-7B \
  --output_dir outputs/inside-lora \
  --report_to none
```

W&B and Hugging Face Hub uploads are off by default. Enable them explicitly with `--report_to wandb` or `--push_to_hub`.

## 4. Generate Outputs

For a local fine-tuned INSIDE model:

```bash
python data_generation/generate.py \
  --config data_generation/configs/local_exp2.example.json
```

Prompting baselines from the paper used GPT models, but you may substitute any provider/model that supports the same chat-completion style interface:

```bash
# exp2_1: standard CoT
python data_generation/generate.py \
  --config data_generation/configs/api_exp2_1_cot.example.json

# exp2_2: Bloom-inspired CoT
python data_generation/generate.py \
  --config data_generation/configs/api_exp2_2_bloomcot.example.json
```

Generation outputs include `output_synthetic`, `output_gt`, and copied metadata.

## 5. Evaluate

First format generations for evaluation:

```bash
python eval/fidelity_eval/scripts/format.py \
  --input data_examples/generation_synthetic.jsonl \
  --output outputs/generation_synthetic_formatted.jsonl
```

Then compute action-fidelity features:

```bash
python eval/fidelity_eval/scripts/add_metrics.py \
  --input_dir outputs \
  --output_dir outputs/features \
  --file generation_synthetic_formatted.jsonl
```

If you provide doctest files named `{semester}_{question_name}.py`, `add_metrics.py` also computes autograder pass rates:

```bash
python eval/fidelity_eval/scripts/add_metrics.py \
  --input_dir outputs \
  --output_dir outputs/features \
  --test_files_dir path/to/doc_tests \
  --file generation_synthetic_formatted.jsonl
```

For internal-dialogue alignment:

```bash
python eval/think_eval/scripts/main.py \
  --config eval/think_eval/configs/api_judge.example.json \
  --input outputs/generation_synthetic_formatted.jsonl \
  --output outputs/think_judged.jsonl \
  --max_workers 8
```

The main think-alignment score is `think_alignment`: the fraction of synthetic-think claims marked `task2_reflected=true`, meaning the claim is reflected in the real code edit.

## Privacy

This release intentionally excludes private student data. To reproduce the full experiments, use your own consented student interaction data in the example schema.

## License

This code is released under the MIT License. The license applies to the released code and synthetic examples only; private course data is not included or licensed for release.

## Citation

Citation information will be added when the paper metadata is finalized.
