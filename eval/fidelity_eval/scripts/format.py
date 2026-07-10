import argparse
import json
import os
import re
from collections import defaultdict


def extract_code_block(text, is_gt=False):
    if not isinstance(text, str):
        return ""
    text = text.replace("<SUBMIT>", "").strip()
    match = re.search(r"<code>\s*(.*?)\s*</code>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    text_without_think = re.sub(
        r"<think>\s*.*?\s*</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()
    if is_gt:
        return text_without_think
    return ""


def extract_thought_block(text):
    if not isinstance(text, str):
        return ""
    for tag in ["think", "student"]:
        match = re.search(
            rf"<{tag}>\s*(.*?)\s*</{tag}>",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()
    return ""


def main():
    parser = argparse.ArgumentParser(
        description="Format generation JSONL for fidelity and think evaluations."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.input) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    grouped = defaultdict(list)
    for row in rows:
        key = (row.get("student_id"), row.get("question_name"))
        grouped[key].append(row)

    output = []
    for group in grouped.values():
        group.sort(key=lambda r: len(r.get("curr_problem_prior_submissions") or []))
        for idx, row in enumerate(group, start=1):
            row = dict(row)
            row["block_num"] = idx
            row["is_processed"] = bool(row.get("output_synthetic") and row.get("output_gt"))
            row["gt_code_block"] = extract_code_block(row.get("output_gt", ""), is_gt=True)
            row["synthetic_code_block"] = extract_code_block(row.get("output_synthetic", ""), is_gt=False)
            row["synthetic_thought"] = extract_thought_block(row.get("output_synthetic", ""))
            row["is_submitted_gt"] = "<SUBMIT>" in row.get("output_gt", "")
            row["is_submitted_syn"] = "<SUBMIT>" in row.get("output_synthetic", "")
            row.pop("input", None)
            output.append(row)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        for row in output:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(output)} formatted rows to {args.output}")


if __name__ == "__main__":
    main()
