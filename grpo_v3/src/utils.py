import random, numpy as np, torch

class EMAVar:
    def __init__(self, tau=0.95):
        self.tau = tau
        self.mean = None
        self.m2 = None

    def update(self, x: torch.Tensor):
        m = x.mean().item()
        v = x.var(unbiased=False).item()
        if self.mean is None:
            self.mean, self.m2 = m, v
        else:
            self.mean = self.tau * self.mean + (1 - self.tau) * m
            self.m2 = self.tau * self.m2 + (1 - self.tau) * v
        return max(self.m2, 1e-6)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_bnb_4bit(bfloat16=True):
    from transformers import BitsAndBytesConfig
    import torch
    compute_dtype = torch.bfloat16 if bfloat16 else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )
