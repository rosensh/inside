########################################################
# TEMPLATES FOR FINE-TUNED MODELS
########################################################

########################################################
# EXP 1: attempt level with received feedback in prior submissions
########################################################

EXP_1_INPUT_TEMPLATE = """PROBLEM INSTRUCTIONS:
{instructions}

FIXED CODE:
<code>{fixed_code}</code>

TODO CODE:
<code>{skeleton_code}</code>

SUBMISSION HISTORY (CODE + FEEDBACK):
{submissions_with_feedback}
"""

########################################################
# EXP 2: same input context, but output = dialogue + code
########################################################

EXP_2_INPUT_TEMPLATE = EXP_1_INPUT_TEMPLATE

########################################################
# TEMPLATES FOR PROMPTING (NON-FINETUNED)
########################################################

########################################################
# EXP 1: attempt level with received feedback in prior submissions
########################################################

EXP_1_INPUT_TEMPLATE_NFT = """PROBLEM INSTRUCTIONS:
{instructions}

FIXED CODE:
<code>{fixed_code}</code>

TODO CODE:
<code>{skeleton_code}</code>

SUBMISSION HISTORY (CODE + FEEDBACK):
{submissions_with_feedback}

YOUR TASK:
You are simulating a student taking an introduction to Python programming course.
Generate the student's next attempt at the TODO CODE of the current problem, using both their prior submissions and the feedback they received.
* When writing the code, you must wrap the generated code in <code> and </code> tags.
* Only write the contents of the TODO CODE. Do not include docstrings or anything outside it. 
* As a novice student, your attempts may include mistakes or partial fixes. 
* FIXED CODE is provided — do not copy, modify, or include it.
* Use prior feedback to inform your code revision.
* If the generated attempt is the student's final submission, append "<SUBMIT>" at the end.
* Use Python programming language
"""

########################################################
# EXP 2_1: CoT prompting
########################################################

EXP_2_1_INPUT_TEMPLATE_NFT = """PROBLEM INSTRUCTIONS:
{instructions}

FIXED CODE:
<code>{fixed_code}</code>

TODO CODE:
<code>{skeleton_code}</code>

SUBMISSION HISTORY (CODE + FEEDBACK):
{submissions_with_feedback}

YOUR TASK:
You are simulating a student taking an introduction to Python programming course.
Generate the student's next attempt at the TODO CODE of the current problem, using both their prior submissions and the feedback they received.

Your task has two steps.

### STEP 1: Generate the student's internal dialogue before their next submission using <think> and </think> tags.

<think>
Write the student's internal dialogue in first-person voice. Keep it to 1–2 sentences.
The dialogue should reflect what the student likely thought before producing the subsequent submission (i.e., the "voice in their head"). It should read like a brief think-aloud explanation of their reasoning.
The thought should sound like a novice programmer thinking through the problem: informal, tentative, and sometimes incomplete. It may include uncertainty, guesses, or small misunderstandings, and may occasionally include questions the student asks themselves.
</think>

### STEP 2: Generate the student's next code submission.
* Use the internal dialogue above to inform the code the student writes next.
* When writing the code, you must wrap the generated code in <code> and </code> tags.
* Only write the contents of the TODO CODE. Do not include docstrings or anything outside it.
* As a novice student, your attempts may include mistakes or partial fixes.
* FIXED CODE is provided — do not copy, modify, or include it.
* Use prior feedback to inform your code revision.
* If the generated attempt is the student's final submission, append "<SUBMIT>" at the end.
* Use Python programming language
"""


########################################################
# EXP 2_2: Bloom-inspired structured CoT prompting
########################################################

EXP_2_2_INPUT_TEMPLATE_NFT = """PROBLEM INSTRUCTIONS:
{instructions}

FIXED CODE:
<code>{fixed_code}</code>

TODO CODE:
<code>{skeleton_code}</code>

SUBMISSION HISTORY (CODE + FEEDBACK):
{submissions_with_feedback}

YOUR TASK:
You are simulating a student taking an introduction to Python programming course.
Generate the student's next attempt at the TODO CODE of the current problem, using both their prior submissions and the feedback they received.

Your task has three steps.

### STEP 1: Infer the student's internal state right after receiving the most recent feedback.

Write these fields from a third-person analyst perspective describing the student, not in the student's own voice.
The states correspond to three learning dimensions commonly used in educational theory: cognitive (knowledge and reasoning), affective (emotions and attitudes toward learning), and action (the concrete problem-solving step the student is about to take).

Use exactly these tags:

<cognitive>
In one sentence, describe the student's current understanding of the problem or feedback. This may include recalling relevant concepts, interpreting the feedback, applying a rule, analyzing the cause of a bug, or forming a hypothesis about how to fix the code. Include both correct and incorrect beliefs or misconceptions suggested by the code changes.
</cognitive>

<affective>
In one sentence, describe the student's emotional or motivational state toward the task, such as confusion, uncertainty, frustration, confidence, curiosity, or persistence. This reflects how the student is reacting to the feedback and the difficulty of the problem.
</affective>

<action>
In one sentence, describe the concrete programming step the student is about to take next. This should reflect the student's problem-solving action.
</action>

### STEP 2: Generate the student's internal dialogue before their next submission using <think> and </think> tags.

<think>
Write the student's internal dialogue in first-person voice, based on the inferred states above (cognitive, affective, action). Keep it to 1–2 sentences.
The dialogue should reflect what the student likely thought before producing the subsequent submission (i.e., the "voice in their head"). It should read like a brief think-aloud explanation of their reasoning.
The thought should sound like a novice programmer thinking through the problem: informal, tentative, and sometimes incomplete. It may include uncertainty, guesses, or small misunderstandings, and may occasionally include questions the student asks themselves.
</think>

### STEP 3: Generate the student's next code submission.
* Use the internal dialogue above to inform the code the student writes next.
* When writing the code, you must wrap the generated code in <code> and </code> tags.
* Only write the contents of the TODO CODE. Do not include docstrings or anything outside it.
* As a novice student, your attempts may include mistakes or partial fixes.
* FIXED CODE is provided — do not copy, modify, or include it.
* Use prior feedback to inform your code revision.
* If the generated attempt is the student's final submission, append "<SUBMIT>" at the end.
* Use Python programming language
"""
