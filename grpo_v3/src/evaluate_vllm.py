import argparse, json, pathlib, os
from typing import List

from tqdm import tqdm
from vllm import LLM, SamplingParams

from parsing import exact_match


def load_data(n_eval: int) -> List[dict]:
    data_path = pathlib.Path("data/gsm8k_test.jsonl")
    rows = [json.loads(x) for x in data_path.read_text(encoding="utf-8").splitlines()][:n_eval]
    return rows


def build_llm(args):
    # vLLM handles tokenizer/model loading internally
    llm = LLM(
        model=args.ckpt_path,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        # HF_TOKEN can be picked up from env for private models
    )
    return llm


def run_eval(args):
    rows = load_data(args.n_eval)
    prompts = [r["prompt"] for r in rows]
    golds = [r["answer"] for r in rows]
    questions = [r.get("question", "") for r in rows]

    log_f = None
    if args.log_path:
        log_path = pathlib.Path(args.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_f = log_path.open("w", encoding="utf-8")

    llm = build_llm(args)

    # Greedy (Pass@1)
    sp_greedy = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
    )
    greedy_outputs = llm.generate(prompts, sampling_params=sp_greedy, use_tqdm=True)

    greedy_correct = 0
    for out, gold, q, prompt, idx in zip(greedy_outputs, golds, questions, prompts, range(len(prompts))):
        text = out.outputs[0].text  # single greedy output
        greedy_correct += exact_match(text, gold)
        if log_f is not None:
            log_entry = {
                "index": idx,
                "mode": "greedy",
                "prompt": prompt,
                "question": q,
                "gold_answer": gold,
                "model_output": text,
            }
            log_f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    pass1 = greedy_correct / len(prompts)

    # Sampling (Pass@k, Maj@k)
    solved_any = 0
    maj_win = 0
    if args.k > 0:
        sp_sample = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=args.max_new_tokens,
            n=args.k,
        )
        sample_outputs = llm.generate(prompts, sampling_params=sp_sample, use_tqdm=True)
        for out, gold, q, prompt, idx in zip(sample_outputs, golds, questions, prompts, range(len(prompts))):
            answers = [o.text for o in out.outputs]
            hits = sum(exact_match(a, gold) for a in answers)
            solved_any += 1 if hits > 0 else 0

            # Majority@k over numeric answers extracted
            import re, collections

            nums = []
            for a in answers:
                m = re.search(r"([\-]?\d+(?:\.\d+)?)", a)
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
                    "prompt": prompt,
                    "question": q,
                    "gold_answer": gold,
                    "samples": answers,
                }
                log_f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    if log_f is not None:
        log_f.close()

    results = {"Pass@1": pass1}
    if args.k > 0:
        results[f"Pass@{args.k}"] = solved_any / len(prompts)
        results[f"Maj@{args.k}"] = maj_win / len(prompts)

    print("\n== Metrics ==")
    for k_, v in results.items():
        print(f"{k_}: {v:.4f}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_path", type=str, required=True, help="HF model name or local checkpoint dir")
    ap.add_argument("--k", type=int, default=8, help="number of stochastic samples per problem")
    ap.add_argument("--n_eval", type=int, default=200, help="number of problems to evaluate")
    ap.add_argument("--max_new_tokens", type=int, default=256, help="generation length")
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--tensor_parallel_size", type=int, default=1, help="tensor parallel size for vLLM")
    ap.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="vLLM GPU memory utilization fraction (0-1)",
    )
    ap.add_argument(
        "--log_path",
        type=str,
        default=None,
        help="optional jsonl log file to store questions, gold answers and model outputs",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
