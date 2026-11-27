"""Prompt templates for EMAIL spam (adapted from the SpaLLM-Guard prompt, Salman et al. 2025, to emails).
Variant 0 is the wording used in all reported runs; variants 1-2 are the alternative wordings for the robustness check."""

SYSTEMS = [
    ("Act as an email spam classifier. Your task is to classify whether a given email is "
     "'Spam' or 'Ham' (legitimate). Do not reprint the email or provide any explanation; "
     "simply respond with 'Spam' for spam emails or 'Ham' for legitimate emails."),
    ("You filter incoming email. Read the email and decide if it is unsolicited bulk or malicious mail ('Spam') "
     "or a normal message the recipient wants ('Ham'). Answer with exactly one word: 'Spam' or 'Ham'."),
    ("Classify the following email as 'Spam' or 'Ham'. 'Ham' means a legitimate email. "
     "Reply only with the label."),
]
ZS_SYSTEM = SYSTEMS[0]
HEADERS = ["Below are some examples of 'spam' and 'ham' emails:", "Labeled examples:", "Examples:"]


def format_email(subject, body, max_chars):
    """One-line text 'Subject: ... | Body: ...' cut to max_chars (keeps prompts short)."""
    s = " ".join(str(subject).split()); b = " ".join(str(body).split())
    return f"Subject: {s} | Body: {b}"[:max_chars]


def build_system_prompt(examples, variant=0):
    """examples: list of (text, label), label 1 = spam, 0 = ham. Empty list -> zero-shot prompt."""
    if not examples:
        return SYSTEMS[variant]
    lines = [SYSTEMS[variant], HEADERS[variant]]
    for i, (t, y) in enumerate(examples):
        t = " ".join(str(t).replace("'", "’").split())
        lead = "For example: the output for" if i == 0 else "The output for"
        lines.append(f"{lead} '{t}' is {'Spam' if int(y) == 1 else 'Ham'}.")
    return "\n".join(lines)
