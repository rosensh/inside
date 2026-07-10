import json
import re

INTERNAL_DIALOGUE_SYSTEM_PROMPT = """
        You are analyzing a novice student interacting with an AI programming tutor and reconstructing the internal dialogue behind the code edits they make.

        This is a retrospective reconstruction task. You are given:
        - the full past interaction history, including previously inferred student dialogue
        - the most recent submission and the feedback on that submission
        - the subsequent submission 

        Students differ in background, experience, confidence, and problem-solving style. They may misunderstand feedback, partially apply it, over-apply it, or ignore parts of it. They often make small local edits instead of fully restructuring their code.

        Your task has two steps.

        ### STEP 1: Infer the student's internal state from the context.

        Write these fields from a third-person analyst perspective describing the student, not in the student's own voice.

        The states correspond to three learning dimensions commonly used in educational theory: cognitive (knowledge and reasoning), affective (emotions and attitudes toward learning), and action (the concrete problem-solving step the student is taking).

        Use exactly these tags:

        <cognitive>
        In one sentence, describe the student's current understanding of the problem or feedback. This may include recalling relevant concepts, interpreting the feedback, applying a rule, analyzing the cause of a bug, or forming a hypothesis about how to fix the code. Include both correct and incorrect beliefs or misconceptions suggested by the code changes.
        </cognitive>

        <affective>
        In one sentence, describe the student's emotional or motivational state toward the task, such as confusion, uncertainty, frustration, confidence, curiosity, or persistence. This reflects how the student is reacting to the feedback and the difficulty of the problem.
        </affective>

        <action>
        In one sentence, describe the concrete programming step the student appears to take next based on the subsequent submission. This should reflect the student's problem-solving action.
        </action>

        ### STEP 2: Generate the student's internal dialogue.

<think>
Reflect on the cognitive, affective, and action states above, then write the student's internal monologue in first-person voice, thinking out loud *before* the next submission.

The monologue must organically reflect all three states. If the student makes an incorrect edit, put the misconception front and center — it should feel completely obvious and correct to them, invisibly driving the wrong code. If the student makes a correct edit, show what clicked.

Voice: messy, informal, mid-thought. May include "wait...", false starts, or half-convincing themselves of something wrong. Remembe you are a novice novice student learning to code for the first time, not an expert analyzing the problem with perfect clarity. 
Length: 2–4 sentences.
</think>

        Output exactly in this format and nothing else:

        <cognitive>...</cognitive>
        <affective>...</affective>
        <action>...</action>
        <think>...</think>
        """.strip()

def extract_think(text):
    if not text:
        return ""
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()

def build_internal_dialogue_user_prompt(example, dialogue_history):
    inp = example["INPUT"]
    output = example["OUTPUT"]

    subs = inp["curr_problem_prior_submissions"]
    fbs = inp["curr_problem_prior_bot_feedback"]

    # =========================
    # Build past history
    # =========================

    past_blocks = []

    for i in range(len(subs) - 1):
        fb = fbs[i] if i < len(fbs) else "(No feedback yet)"

        block = f"""
--- PAST TURN {i} ---

[Student submission]
{subs[i]}

[Feedback]
{fb}
""".rstrip()

        if i < len(dialogue_history):
            past_think = extract_think(dialogue_history[i])
            block += f"""

[Student internal dialogue]
{past_think}
""".rstrip()

        past_blocks.append(block)

    past_history = "\n\n".join(past_blocks) if past_blocks else "(No past history yet)"

    # =========================
    # Current step
    # =========================

    current_prev_sub = subs[-1]

    if len(fbs) >= len(subs):
        current_fb = fbs[len(subs) - 1]
    else:
        current_fb = "(No feedback yet)"

    # =========================
    # Prompt
    # =========================

    prompt = f"""
        Here is the context for the problem they are working on:

        [Problem instructions]
        {inp["instructions"]}

        [Skeleton code]

        Fixed:
        {inp["skeleton_code_fixed"]}

        Todo:
        {inp["skeleton_code_todo"]}

        ==============================
        PAST INTERACTION HISTORY
        ==============================

        {past_history}

        ==============================
        CURRENT STEP
        ==============================

        [Student submission]
        {current_prev_sub}

        [Feedback the student just received]
        {current_fb}

        [Student's subsequent submission]
        {output}

        """.strip()

    return prompt


def build_internal_dialogue_messages(example, dialogue_history):
    return [
        {"role": "system", "content": INTERNAL_DIALOGUE_SYSTEM_PROMPT},
        {"role": "user", "content": build_internal_dialogue_user_prompt(example, dialogue_history)},
    ]


def build_internal_dialogue_prompt(example, dialogue_history):
    return (
        INTERNAL_DIALOGUE_SYSTEM_PROMPT
        + "\n\n"
        + build_internal_dialogue_user_prompt(example, dialogue_history)
    )
