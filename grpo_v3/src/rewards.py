from typing import List
import re
from parsing import exact_match

def reward_math(pred_texts: List[str], gold_texts: List[str], hit_max: List[bool],
                overlong_penalty: float = -0.2, length_penalty_alpha: float = 0.0):
    """
    Rewards:
    - exact_match -> 1.0
    - format bonus: if a boxed number is present but wrong, give a small reward (0.1) to encourage formatting
    - overlong_penalty applied when generation hits max length
    - optional length_penalty_alpha for long outputs
    """
    format_bonus = 0.1
    boxed_pat = re.compile(r"\\boxed\{\s*([\\-]?\\d+(?:\\.\\d+)?)\s*\}")
    rewards = []
    for pred, gold, is_over in zip(pred_texts, gold_texts, hit_max):
        r = exact_match(pred, gold)
        if r == 0:
            if boxed_pat.search(pred):
                r += format_bonus  # reward formatting even if number is wrong
        if is_over:
            r += overlong_penalty
        if length_penalty_alpha > 0:
            L = max(1, len(pred.split()))
            r -= length_penalty_alpha * (L ** 0.5) / 20.0
        rewards.append(float(r))
    return rewards

def normalize_group_advantages(rewards_group: List[float], eps: float = 1e-8):
    import numpy as np
    r = np.array(rewards_group, dtype=float)
    adv = r - r.mean()
    std = r.std() + eps
    return (adv / std).tolist()
