import re
from templates import *

MAX_SUBMISSIONS = 10

def process_instructions(instructions):
    instructions = re.sub(r"\n+", "\n", instructions)
    return instructions.strip()

########################################################
# EXP 1: Generate the next code submission without internal dialogue.
########################################################

def process_input_exp1(example, prompt_template="ft"):
    input = example["INPUT"]
    submissions = input.get("curr_problem_prior_submissions", [])[-MAX_SUBMISSIONS:]
    feedbacks = input.get("curr_problem_prior_bot_feedback", [])[-MAX_SUBMISSIONS:]

    # Choose template
    template = EXP_1_INPUT_TEMPLATE if prompt_template == "ft" else EXP_1_INPUT_TEMPLATE_NFT

    # Zip and format submission-feedback pairs
    pairs = [
        f"<code>{s.strip()}</code>\nFeedback: {f.strip()}"
        for s, f in zip(submissions, feedbacks)
    ]

    processed_input = template.format(
        instructions=process_instructions(input["instructions"]),
        fixed_code=input["skeleton_code_fixed"].strip(),
        skeleton_code=input["skeleton_code_todo"].strip(),
        submissions_with_feedback="\n\n".join(pairs)
    )
    return {"input": processed_input}

def process_output_exp1(example):
    return {"output": f"<code>{example['OUTPUT'].strip()}</code>"}


########################################################
# EXP 2: Generate internal dialogue followed by code.
########################################################

def process_input_exp2(example, prompt_template="ft"):
    if prompt_template == "ft":
        input = example["INPUT"]
        submissions = input.get("curr_problem_prior_submissions", [])[-MAX_SUBMISSIONS:]
        feedbacks = input.get("curr_problem_prior_bot_feedback", [])[-MAX_SUBMISSIONS:]
        pairs = [
            f"<code>{s.strip()}</code>\nFeedback: {f.strip()}"
            for s, f in zip(submissions, feedbacks)
        ]
        processed_input = EXP_2_INPUT_TEMPLATE.format(
            instructions=process_instructions(input["instructions"]),
            fixed_code=input["skeleton_code_fixed"].strip(),
            skeleton_code=input["skeleton_code_todo"].strip(),
            submissions_with_feedback="\n\n".join(pairs)
        )
        return {"input": processed_input}
    return process_input_exp2_2(example)


def process_input_exp2_1(example):
    input = example["INPUT"]
    submissions = input.get("curr_problem_prior_submissions", [])[-MAX_SUBMISSIONS:]
    feedbacks = input.get("curr_problem_prior_bot_feedback", [])[-MAX_SUBMISSIONS:]

    pairs = [
        f"<code>{s.strip()}</code>\nFeedback: {f.strip()}"
        for s, f in zip(submissions, feedbacks)
    ]

    processed_input = EXP_2_1_INPUT_TEMPLATE_NFT.format(
        instructions=process_instructions(input["instructions"]),
        fixed_code=input["skeleton_code_fixed"].strip(),
        skeleton_code=input["skeleton_code_todo"].strip(),
        submissions_with_feedback="\n\n".join(pairs)
    )
    return {"input": processed_input}


def process_input_exp2_2(example):
    input = example["INPUT"]
    submissions = input.get("curr_problem_prior_submissions", [])[-MAX_SUBMISSIONS:]
    feedbacks = input.get("curr_problem_prior_bot_feedback", [])[-MAX_SUBMISSIONS:]

    pairs = [
        f"<code>{s.strip()}</code>\nFeedback: {f.strip()}"
        for s, f in zip(submissions, feedbacks)
    ]

    processed_input = EXP_2_2_INPUT_TEMPLATE_NFT.format(
        instructions=process_instructions(input["instructions"]),
        fixed_code=input["skeleton_code_fixed"].strip(),
        skeleton_code=input["skeleton_code_todo"].strip(),
        submissions_with_feedback="\n\n".join(pairs)
    )
    return {"input": processed_input}

def process_output_exp2(example):
    return {"output": example["OUTPUT"].strip()}
