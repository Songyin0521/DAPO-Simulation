import datasets, json, pathlib

PROMPT_TMPL = r"""You are a careful math tutor. Think step by step and show your full reasoning.
At the very end, add one line exactly as:
\boxed{{<number>}}

Question: {question}
"""

def main():
    ds = datasets.load_dataset("gsm8k", "main")
    out = pathlib.Path("data")
    out.mkdir(exist_ok=True, parents=True)
    for split in ["train", "test"]:
        rows = []
        for ex in ds[split]:
            # 清洗问题/答案中的换行和罕见分隔符，避免 jsonl 被意外拆行
            q = ex["question"].replace("\r", " ").replace("\n", " ").replace("\u2028", " ").replace("\u2029", " ").strip()
            a = ex["answer"].replace("\r", " ").replace("\n", " ").replace("\u2028", " ").replace("\u2029", " ").strip()
            rows.append({
                "question": q,
                "answer": a,
                "prompt": PROMPT_TMPL.format(question=q)
            })
        (out / f"gsm8k_{split}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    print("Prepared: data/gsm8k_train.jsonl, data/gsm8k_test.jsonl")

if __name__ == "__main__":
    main()
