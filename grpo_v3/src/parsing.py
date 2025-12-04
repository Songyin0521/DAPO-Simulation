import re

# Match LaTeX-style `\boxed{<number>}` only
ANS_PAT = re.compile(r"\\boxed\{\s*([\-]?\d+(?:\.\d+)?)\s*\}")

def extract_numeric_answer(text: str):
    m = ANS_PAT.search(text)
    if not m:
        return None
    s = m.group(1)
    try:
        if "." in s:
            return float(s)
        return int(s)
    except Exception:
        return None

def exact_match(pred, gold):
    nums = re.findall(r"([\-]?\d+(?:\.\d+)?)", gold)
    g = None
    if nums:
        try:
            g = float(nums[-1]) if "." in nums[-1] else int(nums[-1])
        except Exception:
            pass
    p = extract_numeric_answer(pred)
    if p is None or g is None:
        return 0.0
    return 1.0 if float(p) == float(g) else 0.0
