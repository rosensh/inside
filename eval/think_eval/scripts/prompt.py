import difflib
import re

ALIGNMENT_SYSTEM_PROMPT = """You are evaluating a model that simulates a novice student working on a Python programming assignment.

## Your Task

Extract each distinct claim or intention from the Synthetic Think (e.g., "I will add a base case", "I'll use n % 10"). For each claim, evaluate it against both diffs:

- **Task 1** — Does the claim appear in the **synthetic code diff** (t -> synthetic t+1)?
- **Task 2** — Does the same claim appear in the **GT code diff** (t -> GT t+1)?

If a claim is reflected in both, it suggests the synthetic think captures the same intent as the ground truth.

## Output Format

Respond in this exact JSON format:
{{
  "claims": [
    {{
      "claim": "<claim extracted from synthetic think>",
      "task1_rationale": "<why this claim is/isn't reflected in the synthetic diff>",
      "task1_reflected": true or false,
      "task2_rationale": "<why this claim is/isn't reflected in the GT diff>",
      "task2_reflected": true or false
    }},
    {{
      "claim": "<claim extracted from synthetic think>",
      "task1_rationale": "<why this claim is/isn't reflected in the synthetic diff>",
      "task1_reflected": true or false,
      "task2_rationale": "<why this claim is/isn't reflected in the GT diff>",
      "task2_reflected": true or false
    }}
  ]
}}
""".strip()


def compute_diff(code_t, code_t1):
    """Return a unified diff string between code_t and code_t1."""
    before = (code_t or "").splitlines(keepends=True)
    after = (code_t1 or "").splitlines(keepends=True)
    diff = list(difflib.unified_diff(before, after, fromfile="code_t", tofile="code_t1"))
    return "".join(diff) if diff else "(no changes)"


def extract_think(text):
    if not isinstance(text, str):
        return None
    match = re.search(r"<think>(.*?)</think>", text, re.S | re.I)
    return match.group(1).strip() if match else None


def build_alignment_user_prompt(
    code_t,
    feedback_t,
    gt_think_t,
    gt_code_t1,
    syn_think_t,
    syn_code_t1,
    instructions="",
):
    gt_diff = compute_diff(code_t, gt_code_t1)
    syn_diff = compute_diff(code_t, syn_code_t1)
    return f"""## Context

### Problem Description:
{instructions}

### Student's Code at Time t (most recent prior submission):
{code_t}

### Feedback the Student Received at Time t:
{feedback_t}

### Ground Truth Think at Time t (what a real student likely thought):
{gt_think_t}

### Ground Truth Code at Time t+1 (what the real student submitted next):
{gt_code_t1}

### Synthetic Think at Time t (what the model thinks the student thought):
{syn_think_t}

### Synthetic Code at Time t+1 (what the model predicts the student will submit next):
{syn_code_t1}

### Code Diff (t -> synthetic t+1):
{syn_diff}

### Code Diff (t -> GT t+1):
{gt_diff}
""".strip()


def build_alignment_messages(
    code_t,
    feedback_t,
    gt_think_t,
    gt_code_t1,
    syn_think_t,
    syn_code_t1,
    instructions="",
):
    return [
        {"role": "system", "content": ALIGNMENT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_alignment_user_prompt(
                code_t,
                feedback_t,
                gt_think_t,
                gt_code_t1,
                syn_think_t,
                syn_code_t1,
                instructions=instructions,
            ),
        },
    ]


def build_alignment_prompt(
    code_t,
    feedback_t,
    gt_think_t,
    gt_code_t1,
    syn_think_t,
    syn_code_t1,
    instructions="",
):
    return (
        ALIGNMENT_SYSTEM_PROMPT
        + "\n\n"
        + build_alignment_user_prompt(
            code_t,
            feedback_t,
            gt_think_t,
            gt_code_t1,
            syn_think_t,
            syn_code_t1,
            instructions=instructions,
        )
    )
