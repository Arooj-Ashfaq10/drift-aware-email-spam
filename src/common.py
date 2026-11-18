"""Prompt template for EMAIL spam (adapted from the SpaLLM-Guard prompt, Salman et al. 2025, to emails)."""

ZS_SYSTEM = (
    "Act as an email spam classifier. Your task is to classify whether a given email is "
    "'Spam' or 'Ham' (legitimate). Do not reprint the email or provide any explanation; "
    "simply respond with 'Spam' for spam emails or 'Ham' for legitimate emails."
)


def format_email(subject, body, max_chars):
    """One-line text 'Subject: ... | Body: ...' cut to max_chars (keeps prompts short)."""
    s = " ".join(str(subject).split()); b = " ".join(str(body).split())
    return f"Subject: {s} | Body: {b}"[:max_chars]


def build_system_prompt(examples):
    """examples: list of (text, label), label 1 = spam, 0 = ham. Empty list -> zero-shot prompt."""
    if not examples:
        return ZS_SYSTEM
    lines = [ZS_SYSTEM, "Below are some examples of 'spam' and 'ham' emails:"]
    for i, (t, y) in enumerate(examples):
        t = " ".join(str(t).replace("'", "’").split())
        lead = "For example: the output for" if i == 0 else "The output for"
        lines.append(f"{lead} '{t}' is {'Spam' if int(y) == 1 else 'Ham'}.")
    return "\n".join(lines)
