import json, pathlib, random

class GSM8KJsonl:
    def __init__(self, path: str):
        self.rows = [json.loads(x) for x in pathlib.Path(path).read_text(encoding="utf-8").splitlines()]
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]

def load_jsonl(path): return GSM8KJsonl(path)

def batchify_indices(n, bs):
    idx = list(range(n))
    random.shuffle(idx)
    for i in range(0, n, bs):
        yield idx[i:i+bs]
