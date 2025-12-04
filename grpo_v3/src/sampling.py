from typing import List
import torch

@torch.no_grad()
def generate_k(model, tokenizer, prompts: List[str], k: int, max_new_tokens: int, temperature: float = 0.7):
    device = next(model.parameters()).device
    B = len(prompts)
    texts, hit_max = [], []
    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt").to(device)
        out = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            num_return_sequences=k,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        for seq in out:
            txt = tokenizer.decode(seq, skip_special_tokens=True)
            texts.append(txt)
            hit_max.append(txt.strip().endswith(tokenizer.eos_token) is False)
    batched = [texts[i*k:(i+1)*k] for i in range(B)]
    batched_over = [hit_max[i*k:(i+1)*k] for i in range(B)]
    return batched, batched_over
