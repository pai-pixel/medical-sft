import re
from typing import List
from swift.rewards.orm import ORM, orms

# Our data prompt forces the model to end with 「答案：X」, X in A-D.
_CHOICE = re.compile(r'答案\s*[:：]\s*([A-Da-d])')
_FALLBACK = re.compile(r'([A-Da-d])')


def _to_text(c):
    # swift may pass a completion as str, dict, or list-of-dicts; normalize to text.
    if isinstance(c, str):
        return c
    if isinstance(c, list) and c and isinstance(c[-1], dict):
        return c[-1].get('content', '')
    if isinstance(c, dict):
        return c.get('content', '')
    return str(c)


def _extract(text: str):
    text = text or ''
    m = _CHOICE.findall(text)        # primary: the 「答案：X」 the prompt asked for
    if m:
        return m[-1].upper()
    tail = text[-120:]               # fallback: last bare A-D near the end
    m = _FALLBACK.findall(tail)
    return m[-1].upper() if m else None


class MCQAccuracy(ORM):
    """1.0 if extracted choice letter == gold solution, else 0.0."""
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        out = []
        for c, sol in zip(completions, solution):
            pred = _extract(_to_text(c))
            out.append(1.0 if pred and pred == str(sol).strip().upper() else 0.0)
        return out


class MCQFormat(ORM):
    """1.0 if the completion contains a well-formed 「答案：X」, else 0.0."""
    def __call__(self, completions, **kwargs) -> List[float]:
        return [1.0 if _CHOICE.search(_to_text(c)) else 0.0 for c in completions]


orms['mcq_acc'] = MCQAccuracy
orms['mcq_format'] = MCQFormat
