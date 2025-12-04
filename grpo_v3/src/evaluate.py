
import argparse, json, pathlib, os
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from parsing import exact_match


@torch.no_grad()
def gen(model, tok, prompt, max_new_tokens=256, temperature=None):
    device = next(model.parameters()).device
    inputs = tok(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]
    out = model.generate(
        **inputs,
        do_sample=temperature is not None,
        temperature=temperature if temperature is not None else 1.0,
        top_p=0.95 if temperature is not None else None,
        max_new_tokens=max_new_tokens,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    gen_tokens = out[0][input_len:]
    return tok.decode(gen_tokens, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_path", type=str, required=True, help="HF model name or local checkpoint dir")
    ap.add_argument("--k", type=int, default=8, help="number of stochastic samples per problem")
    ap.add_argument("--n_eval", type=int, default=200, help="number of problems to evaluate")
    ap.add_argument("--max_new_tokens", type=int, default=256, help="generation length")
    ap.add_argument("--quantize4bit", action="store_true", help="load model in 4-bit to save VRAM")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    ap.add_argument(
        "--log_path",
        type=str,
        default=None,
        help="optional jsonl log file to store questions, gold answers and model outputs",
    )
    args = ap.parse_args()

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(
        args.ckpt_path, use_fast=True, trust_remote_code=True, token=os.getenv("HF_TOKEN", None)
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Model (4-bit optional)
    quant = None
    if args.quantize4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float16,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt_path,
        device_map="auto",
        quantization_config=quant,
        trust_remote_code=True,
        token=os.getenv("HF_TOKEN", None),
    )
    model.eval()

    # Dataset
    data_path = pathlib.Path("data/gsm8k_test.jsonl")
    rows = [json.loads(x) for x in data_path.read_text(encoding="utf-8").splitlines()][: args.n_eval]

    log_f = None
    if args.log_path:
        log_path = pathlib.Path(args.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_f = log_path.open("w", encoding="utf-8")

    # Pass@1 (greedy)
    greedy_correct = 0
    for idx, r in enumerate(tqdm(rows, desc="Pass@1 (greedy)", unit="q")):
        p = r["prompt"]
        gold = r["answer"]
        question = r.get("question", "")
        out = gen(model, tok, p, temperature=None, max_new_tokens=args.max_new_tokens)
        greedy_correct += exact_match(out, gold)
        if log_f is not None:
            log_entry = {
                "index": idx,
                "mode": "greedy",
                "prompt": p,
                "question": question,
                "gold_answer": gold,
                "model_output": out,
            }
            log_f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    pass1 = greedy_correct / len(rows)

    # Stochastic sampling
    solved_any = 0
    maj_win = 0
    if args.k > 0:
        for idx, r in enumerate(tqdm(rows, desc=f"Sampling (k={args.k})", unit="q")):
            p = r["prompt"]
            gold = r["answer"]
            question = r.get("question", "")
            answers = [
                gen(model, tok, p, temperature=0.7, max_new_tokens=args.max_new_tokens) for _ in range(args.k)
            ]
            hits = sum(exact_match(a, gold) for a in answers)
            solved_any += 1 if hits > 0 else 0

            # Majority@k
            import re, collections

            nums = []
            for a in answers:
                m = re.search(r"Final Answer:\s*([\-]?\d+(?:\.\d+)?)", a)
                if m:
                    nums.append(m.group(1))
            if nums:
                c = collections.Counter(nums)
                maj = c.most_common(1)[0][0]
                g_nums = re.findall(r"([\-]?\d+(?:\.\d+)?)", gold)
                if g_nums and maj == g_nums[-1]:
                    maj_win += 1

            if log_f is not None:
                log_entry = {
                    "index": idx,
                    "mode": f"sampling_k={args.k}",
                    "prompt": p,
                    "question": question,
                    "gold_answer": gold,
                    "samples": answers,
                }
                log_f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    if log_f is not None:
        log_f.close()

    k = args.k
    results = {"Pass@1": pass1}
    if k > 0:
        results[f"Pass@{k}"] = solved_any / len(rows)
        results[f"Maj@{k}"] = maj_win / len(rows)

    print("\n== Metrics ==")
    for k_, v in results.items():
        print(f"{k_}: {v:.4f}")


if __name__ == "__main__":
    main()
