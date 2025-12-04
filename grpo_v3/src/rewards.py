from typing import List
import re
from parsing import exact_match

def reward_math(
    pred_texts: List[str],
    gold_texts: List[str],
    hit_max: List[bool],
    overlong_penalty: float = -0.2,
    length_penalty_alpha: float = 0.0,
    max_new_tokens: int = None,
    tokenizer=None,
):
    """
    Rewards:
    - exact_match -> 1.0
    - format bonus: if a boxed number is present but wrong, give a small reward (0.1) to encourage formatting
    - overlong_penalty applied when generation hits max length
    - progressive length penalty: starts at 0.8 * max_new_tokens, reaches full overlong_penalty at 1.2 * max_new_tokens
    - optional length_penalty_alpha for long outputs
    """
    format_bonus = 0.1
    boxed_pat = re.compile(r"\\boxed\{\s*([\\-]?\\d+(?:\\.\\d+)?)\s*\}")
    rewards = []
    for pred, gold, is_over in zip(pred_texts, gold_texts, hit_max):
        r = exact_match(pred, gold)
        if r == 0 and boxed_pat.search(pred):
            r += format_bonus  # reward formatting even if number is wrong

        # Fixed penalty when hitting max length
        if is_over:
            r += overlong_penalty

        # Progressive penalty beyond 0.8 * max_new_tokens up to 1.2 * max_new_tokens
        if max_new_tokens and max_new_tokens > 0:
            if tokenizer is not None:
                L = len(tokenizer(pred, add_special_tokens=False).input_ids)
            else:
                L = len(pred.split())
            ratio = L / float(max_new_tokens)
            if ratio > 0.8:
                frac = min(1.0, (ratio - 0.8) / 0.4)  # 0 at 0.8x, 1 at 1.2x
                r += overlong_penalty * frac

        if length_penalty_alpha > 0:
            Lw = max(1, len(pred.split()))
            r -= length_penalty_alpha * (Lw ** 0.5) / 20.0
        rewards.append(float(r))
    return rewards

def normalize_group_advantages(rewards_group: List[float], eps: float = 1e-8):
    import numpy as np
    r = np.array(rewards_group, dtype=float)
    adv = r - r.mean()
    std = r.std() + eps
    return (adv / std).tolist()
