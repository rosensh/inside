########################################################
# EXP 1 and EXP 2 use the same input context.
# EXP 1 outputs code only; EXP 2 outputs internal dialogue + code.
########################################################

EXP_INPUT_TEMPLATE = """PROBLEM INSTRUCTIONS:
{instructions}

FIXED CODE:
<code>{fixed_code}</code>

TODO CODE:
<code>{skeleton_code}</code>

SUBMISSION HISTORY (CODE + FEEDBACK):
{submissions_with_feedback}
"""
