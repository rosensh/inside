import argparse
import copy
import json
import os
import random
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

from prompt import build_internal_dialogue_messages, build_internal_dialogue_prompt


write_lock = threading.Lock()


def load_config(path):
    with open(path) as f:
        cfg = json.load(f)
    cfg["api_base"] = cfg.get("api_base") or os.getenv("OPENAI_BASE_URL", "")
    cfg["api_key"] = cfg.get("api_key") or os.getenv("OPENAI_API_KEY", "")
    cfg["model"] = cfg.get("model")
    if not cfg["api_key"]:
        raise ValueError("Missing API key. Set api_key in config or OPENAI_API_KEY.")
    if not cfg["model"]:
        raise ValueError("Missing model. Set model in config.")
    return cfg


def create_client(cfg):
    kwargs = {"api_key": cfg["api_key"]}
    if cfg.get("api_base"):
        kwargs["base_url"] = cfg["api_base"]
    return OpenAI(**kwargs)


def read_json_or_jsonl(path):
    with open(path) as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]


def call_with_retry(client, cfg, messages, max_retries=10):
    for attempt in range(1, max_retries + 1):
        try:
            return client.chat.completions.create(
                model=cfg["model"],
                messages=messages,
                max_completion_tokens=cfg.get("max_completion_tokens", 1024),
                reasoning_effort=cfg.get("reasoning_effort", "minimal"),
            )
        except Exception as exc:
            sleep_time = min(2 * attempt, 30)
            print(f"[retry {attempt}/{max_retries}] {exc}; sleeping {sleep_time}s")
            time.sleep(sleep_time)
    raise RuntimeError("Too many API retries.")


def extract_think(dialogue):
    match = re.search(r"<think>(.*?)</think>", dialogue, re.S | re.I)
    return match.group(1).strip() if match else dialogue.strip()


def process_one_trajectory(key, examples, cfg, client, out_f, pbar, prior_history=None):
    dialogue_history = list(prior_history) if prior_history else []

    for ex in examples:
        messages = build_internal_dialogue_messages(ex, dialogue_history)
        response = call_with_retry(client, cfg, messages)
        dialogue = response.choices[0].message.content.strip()
        think_text = extract_think(dialogue)

        ex_new = copy.deepcopy(ex)
        old_output = ex["OUTPUT"]
        ex_new["RAW_INTERNAL_DIALOGUE_OUTPUT"] = dialogue
        ex_new["OUTPUT"] = f"<think>\n{think_text}\n</think>\n\n{old_output}"

        with write_lock:
            out_f.write(json.dumps(ex_new, ensure_ascii=False) + "\n")
            out_f.flush()
            pbar.update(1)

        dialogue_history.append(think_text)


def group_examples(data, question_filter=None):
    groups = defaultdict(list)
    for ex in data:
        inp = ex["INPUT"]
        key = (inp["student_id"], inp["question_name"])
        if question_filter and key[1] not in question_filter:
            continue
        groups[key].append(ex)

    for key in groups:
        groups[key].sort(
            key=lambda row: len(row["INPUT"].get("curr_problem_prior_submissions", []))
        )
    return groups


def load_completed(output_path):
    done = defaultdict(list)
    if not os.path.exists(output_path):
        return done
    with open(output_path) as f:
        for line in f:
            try:
                row = json.loads(line)
                inp = row["INPUT"]
                done[(inp["student_id"], inp["question_name"])].append(row)
            except Exception:
                continue
    for key in done:
        done[key].sort(
            key=lambda row: len(row["INPUT"].get("curr_problem_prior_submissions", []))
        )
    return done


def main():
    parser = argparse.ArgumentParser(
        description="Generate retrospective internal dialogue traces for INSIDE training."
    )
    parser.add_argument("--config", default="config.example.json")
    parser.add_argument("--input", required=True, help="JSON/JSONL examples without traces.")
    parser.add_argument("--output", required=True, help="JSONL file to append traced examples to.")
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--question", action="append", default=None, help="Optional question filter; can be repeated.")
    parser.add_argument("--test", action="store_true", help="Print and run one random trajectory.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    client = create_client(cfg)
    data = read_json_or_jsonl(args.input)
    groups = group_examples(data, question_filter=set(args.question or []))
    if not groups:
        raise ValueError("No trajectories found after filtering.")

    if args.test:
        key = random.choice(list(groups.keys()))
        dialogue_history = []
        print(f"Test trajectory: student={key[0]} question={key[1]} turns={len(groups[key])}")
        for ex in groups[key]:
            prompt = build_internal_dialogue_prompt(ex, dialogue_history)
            print("\n" + "=" * 80)
            print(prompt)
            response = call_with_retry(client, cfg, build_internal_dialogue_messages(ex, dialogue_history))
            dialogue = response.choices[0].message.content.strip()
            print("\nMODEL OUTPUT\n")
            print(dialogue)
            dialogue_history.append(extract_think(dialogue))
        return

    done = load_completed(args.output)
    new_groups = []
    for key, examples in groups.items():
        completed = done.get(key, [])
        if len(completed) >= len(examples):
            continue
        prior_history = [extract_think(row.get("OUTPUT", "")) for row in completed]
        new_groups.append((key, examples[len(completed):], prior_history))

    total_turns = sum(len(v) for v in groups.values())
    done_count = sum(len(v) for v in done.values())
    remaining = sum(len(examples) for _, examples, _ in new_groups)
    print(f"Total turns: {total_turns}; already generated: {done_count}; remaining: {remaining}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "a", encoding="utf-8") as out_f:
        pbar = tqdm(total=total_turns, initial=done_count, desc="Generating dialogues")
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [
                executor.submit(process_one_trajectory, key, examples, cfg, client, out_f, pbar, prior)
                for key, examples, prior in new_groups
            ]
            for future in as_completed(futures):
                future.result()
        pbar.close()


if __name__ == "__main__":
    main()
