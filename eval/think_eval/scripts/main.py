import argparse
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

from prompt import build_alignment_messages, build_alignment_prompt, extract_think


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


def call_llm(client, cfg, messages, max_retries=5):
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=cfg["model"],
                messages=messages,
                max_completion_tokens=cfg.get("max_tokens", 2048),
                reasoning_effort=cfg.get("reasoning_effort", "low"),
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            if getattr(exc, "code", None) == "content_filter" or "content_filter" in str(exc):
                print("[skip] content filter triggered")
                return None
            print(f"[retry {attempt}/{max_retries}] {exc}")
            time.sleep(min(2 * attempt, 30))
    return None


def extract_context(row):
    prior_subs = row.get("curr_problem_prior_submissions") or []
    prior_fbs = row.get("curr_problem_prior_bot_feedback") or []
    return {
        "code_t": prior_subs[-1] if prior_subs else "",
        "feedback_t": prior_fbs[-1] if prior_fbs else "",
        "gt_think_t": row.get("gt_thought") or extract_think(row.get("output_gt", "")) or "",
        "gt_code_t1": row.get("gt_code_block", ""),
        "syn_think_t": row.get("synthetic_thought") or extract_think(row.get("output_synthetic", "")),
        "syn_code_t1": row.get("synthetic_code_block", ""),
        "instructions": row.get("instructions", ""),
    }


def has_required_fields(ctx):
    return bool(ctx["syn_think_t"]) and bool(ctx["gt_code_t1"]) and bool(ctx["syn_code_t1"])


def row_key(row):
    return (
        row.get("student_id"),
        row.get("question_name"),
        row.get("block_num"),
        row.get("model_name"),
        row.get("experiment"),
    )


def process_row(row, client, cfg):
    ctx = extract_context(row)
    messages = build_alignment_messages(
        ctx["code_t"],
        ctx["feedback_t"],
        ctx["gt_think_t"],
        ctx["gt_code_t1"],
        ctx["syn_think_t"],
        ctx["syn_code_t1"],
        instructions=ctx["instructions"],
    )
    raw = call_llm(client, cfg, messages)
    result = {
        "student_id": row.get("student_id"),
        "question_name": row.get("question_name"),
        "block_num": row.get("block_num"),
        "model_name": row.get("model_name"),
        "experiment": row.get("experiment"),
        "syn_think": ctx["syn_think_t"],
        "claims": None,
        "think_alignment": None,
        "raw_response": raw,
    }
    if raw:
        try:
            parsed = json.loads(raw)
            claims = parsed.get("claims") or []
            result["claims"] = claims
            if claims:
                result["think_alignment"] = sum(
                    1 for claim in claims if claim.get("task2_reflected")
                ) / len(claims)
        except json.JSONDecodeError:
            print(f"[warn] JSON parse failed for {row_key(row)}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="LLM-judge evaluation for INSIDE internal-dialogue alignment."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True, help="Formatted JSONL with synthetic_thought and code blocks.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--question", action="append", default=None, help="Optional question filter; can be repeated.")
    parser.add_argument("--experiment", action="append", default=None, help="Optional experiment filter; can be repeated.")
    parser.add_argument("--model", action="append", default=None, help="Optional model filter; can be repeated.")
    parser.add_argument("--sample_rate", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    rows = read_json_or_jsonl(args.input)
    question_filter = set(args.question or [])
    experiment_filter = set(args.experiment or [])
    model_filter = set(args.model or [])
    filtered = []
    for row in rows:
        if question_filter and row.get("question_name") not in question_filter:
            continue
        if experiment_filter and row.get("experiment") not in experiment_filter:
            continue
        if model_filter and row.get("model_name") not in model_filter:
            continue
        if has_required_fields(extract_context(row)):
            filtered.append(row)
    if args.sample_rate < 1.0:
        filtered = [row for row in filtered if random.random() < args.sample_rate]

    print(f"Rows to judge: {len(filtered)} / {len(rows)}")
    if not filtered:
        return

    cfg = load_config(args.config)
    client = create_client(cfg)

    if args.test:
        row = random.choice(filtered)
        ctx = extract_context(row)
        prompt = build_alignment_prompt(
            ctx["code_t"],
            ctx["feedback_t"],
            ctx["gt_think_t"],
            ctx["gt_code_t1"],
            ctx["syn_think_t"],
            ctx["syn_code_t1"],
            instructions=ctx["instructions"],
        )
        print(prompt)
        print("\nRAW RESPONSE\n")
        messages = build_alignment_messages(
            ctx["code_t"],
            ctx["feedback_t"],
            ctx["gt_think_t"],
            ctx["gt_code_t1"],
            ctx["syn_think_t"],
            ctx["syn_code_t1"],
            instructions=ctx["instructions"],
        )
        print(call_llm(client, cfg, messages))
        return

    done = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                if line.strip():
                    done.add(row_key(json.loads(line)))
    filtered = [row for row in filtered if row_key(row) not in done]
    print(f"Remaining after resume: {len(filtered)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "a", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [executor.submit(process_row, row, client, cfg) for row in filtered]
            for future in tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                with write_lock:
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out_f.flush()


if __name__ == "__main__":
    main()
