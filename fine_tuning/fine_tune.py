import argparse
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from custom_trainer import SeparateLossTrainer
from utils import (
    process_input,
    process_output_exp2,
    process_output_exp1,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning for INSIDE student simulators."
    )
    parser.add_argument(
        "--experiment_name",
        choices=["exp1", "exp2"],
        required=True,
        help="exp1: generate code only; exp2: INSIDE internal dialogue followed by code.",
    )
    parser.add_argument("--dataset_file", required=True, help="Training JSON/JSONL file.")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-Coder-7B")
    parser.add_argument("--output_dir", default="outputs/inside-lora")
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num_train_epochs", type=float, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", default="all-linear")
    parser.add_argument("--lora_bias", default="none")

    parser.add_argument("--report_to", default="none", help='Use "wandb" to log to W&B.')
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", default=None)
    parser.add_argument("--hub_private_repo", action="store_true")
    return parser.parse_args()


def prepare_dataset(args):
    dataset = load_dataset("json", data_files=args.dataset_file, split="train")
    if args.experiment_name == "exp1":
        dataset = dataset.map(process_input, load_from_cache_file=False)
        dataset = dataset.map(process_output_exp1, load_from_cache_file=False)
    elif args.experiment_name == "exp2":
        dataset = dataset.map(process_input, load_from_cache_file=False)
        dataset = dataset.map(process_output_exp2, load_from_cache_file=False)
    else:
        raise ValueError(f"Unsupported experiment: {args.experiment_name}")
    return dataset.remove_columns(["INPUT", "OUTPUT"])


def main():
    args = parse_args()
    run_name = args.run_name or f"{args.experiment_name}-{args.model_name.split('/')[-1]}"

    dataset = prepare_dataset(args)
    split = dataset.train_test_split(
        shuffle=True,
        test_size=args.test_size,
        seed=args.seed,
    )
    train_dataset, eval_dataset = split["train"], split["test"]

    print("Example training pair")
    print("-" * 80)
    print(train_dataset[0]["input"])
    print("-" * 80)
    print(train_dataset[0]["output"])
    print("-" * 80)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def preprocess(example):
        input_text = example["input"]
        output_text = example["output"] + tokenizer.eos_token
        tokenized_full = tokenizer(
            input_text + output_text,
            truncation=True,
            max_length=args.max_seq_length,
            return_tensors="pt",
        )
        input_ids_full = tokenized_full["input_ids"][0]
        input_tokenized = tokenizer(
            input_text,
            truncation=True,
            max_length=args.max_seq_length,
            return_tensors="pt",
        )
        split_idx = len(input_tokenized["input_ids"][0])
        labels = [-100] * split_idx + input_ids_full[split_idx:].tolist()

        think_boundary = len(input_ids_full)
        lower_output = output_text.lower()
        if "</think>" in lower_output:
            end = lower_output.index("</think>") + len("</think>")
            input_plus_think = input_text + output_text[:end]
            think_boundary = len(
                tokenizer(
                    input_plus_think,
                    truncation=True,
                    max_length=args.max_seq_length,
                    return_tensors="pt",
                )["input_ids"][0]
            )

        return {
            "input_ids": input_ids_full.tolist(),
            "attention_mask": tokenized_full["attention_mask"][0].tolist(),
            "labels": labels,
            "think_boundary": think_boundary,
        }

    train_cols = train_dataset.column_names
    eval_cols = eval_dataset.column_names
    train_dataset = train_dataset.map(preprocess, batched=False, load_from_cache_file=False)
    train_dataset = train_dataset.remove_columns(train_cols)
    eval_dataset = eval_dataset.map(preprocess, batched=False, load_from_cache_file=False)
    eval_dataset = eval_dataset.remove_columns(eval_cols)

    base_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=None,
        padding="longest",
        max_length=args.max_seq_length,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    def data_collator(features):
        think_boundaries = [f.pop("think_boundary") for f in features]
        batch = base_collator(features)
        batch["think_boundary"] = torch.tensor(think_boundaries, dtype=torch.long)
        return batch

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        bias=args.lora_bias,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        bf16=True,
        tf32=True,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        eval_on_start=True,
        save_strategy="steps",
        save_steps=args.save_steps,
        remove_unused_columns=False,
        report_to=args.report_to,
        run_name=run_name,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_private_repo=args.hub_private_repo,
    )

    trainer_cls = SeparateLossTrainer if args.experiment_name == "exp2" else Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    trainer.train()

    if args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
