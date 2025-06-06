import re

_CLEAN_PATTERNS = [
    (re.compile(r"`` "), r'"'),
    (re.compile(r" ''"), r'"'),
    (re.compile(r"whats ", flags=re.IGNORECASE), r"what's "),
    (re.compile(r'\s+([.,!?;:])'), r'\1'),
]

_APOS_PATTERNS = [
    (re.compile(r"\s+n't"), r"n't"),
    (re.compile(r"\s+'"), r"'"),
]

def clean_text(text: str) -> str:
    text = text.replace(" cant ", " can't ")
    text = text.replace(" wont ", " won't ")
    text = text.replace(" wouldnt ", " wouldn't ")
    text = text.replace(" arent ", " aren't ")
    text = text.replace(" youre ", " you're ")
    text = text.replace(" theyre ", " they're ")
    text = text.replace("( ", "(")
    text = text.replace(" )", ")")

    for pat, repl in _CLEAN_PATTERNS:
        text = pat.sub(repl, text)

    if "'" in text:
        for patt, repl in _APOS_PATTERNS:
            text = patt.sub(repl, text)

    text = text.capitalize()
    text = re.sub(r" i ", r" I ", text)

    if text.startswith('"'):

        if len(text) > 1:
            text = '"' + text[1].upper() + text[2:]

        if text.count('"') == 1:
            text += '"'

    if text.endswith('"') and text.count('"') == 1:
        text = '"' + text

    return text.strip()
