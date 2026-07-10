import re

from templates import EXP_INPUT_TEMPLATE


MAX_SUBMISSIONS = 10


def process_instructions(instructions):
    instructions = re.sub(r"\n+", "\n", instructions)
    return instructions.strip()


def process_curr_problem_history(example, max_submissions=10):
    submissions = example["curr_problem_prior_submissions"][-max_submissions:]
    feedbacks = example.get("curr_problem_prior_bot_feedback", [])[-max_submissions:]
    blocks = []
    for sub, fb in zip(submissions, feedbacks):
        blocks.append(
            f"SUBMISSION:\n<code>{sub.strip()}</code>\nFEEDBACK RECEIVED:\n{fb.strip()}"
        )
    return "\n\n".join(blocks)


def process_input(example):
    input_record = example["INPUT"]
    processed_input = EXP_INPUT_TEMPLATE.format(
        instructions=process_instructions(input_record["instructions"]),
        fixed_code=input_record["skeleton_code_fixed"].strip(),
        skeleton_code=input_record["skeleton_code_todo"].strip(),
        submissions_with_feedback=process_curr_problem_history(input_record, MAX_SUBMISSIONS),
    )
    return {"input": processed_input}


def process_output_exp1(example):
    """Paper Experiment 1: generate the next code submission only."""
    return {"output": f"<code>{example['OUTPUT'].strip()}</code>"}


def process_output_exp2(example):
    """Paper Experiment 2 / INSIDE: generate internal dialogue followed by code."""
    text = example["OUTPUT"].strip()
    if "</think>" in text:
        thought, rest = text.split("</think>", 1)
        thought = thought + "</think>"
        code = rest.strip()
    else:
        thought = ""
        code = text

    wrapped = f"""{thought}

<code>
{code}
</code>
"""
    return {"output": wrapped.strip()}
