"""
Minimal GRPO training script using Hugging Face TRL (GRPOTrainer).
Assumes a local `data/gsm8k_train.jsonl` with fields: {"prompt": ..., "answer": ...}
and a LoRA/4bit friendly base model (e.g., Qwen/Qwen2.5-Math-1.5B).
"""
import argparse, json, pathlib, re
from typing import Dict, List

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer
from parsing import exact_match


def load_jsonl(path: str) -> List[Dict]:
    return [json.loads(x) for x in pathlib.Path(path).read_text(encoding="utf-8").splitlines()]


boxed_pat = re.compile(r"\\boxed\\{\\s*([\\-]?\\d+(?:\\.\\d+)?)\\s*\\}")


def build_reward_fn(max_new_tokens: int = 256):
    def reward_fn(samples: List[str], **kwargs) -> List[float]:
        answers = kwargs["answers"]
        rewards = []
        for pred, gold in zip(samples, answers):
            r = exact_match(pred, gold)  # 1 or 0 based on boxed numeric match
            # format bonus if boxed but wrong
            if r == 0 and boxed_pat.search(pred):
                r += 0.1
            # length penalty when too long (simple linear)
            L = len(pred.split())
            ratio = L / float(max_new_tokens)
            if ratio > 1.0:
                r -= min(0.5, (ratio - 1.0))
            rewards.append(float(r))
        return rewards

    return reward_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-Math-1.5B")
    ap.add_argument("--data_path", default="data/gsm8k_train.jsonl")
    ap.add_argument("--output_dir", default="outputs/grpo_trl")
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--group_size", type=int, default=4)
    ap.add_argument("--per_device_batch_size", type=int, default=1)
    ap.add_argument("--bnb_4bit", action="store_true")
    args = ap.parse_args()

    rows = load_jsonl(args.data_path)
    ds = Dataset.from_list([{"prompt": r["prompt"], "answer": r["answer"]} for r in rows])

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_cfg = None
    if args.bnb_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quant_cfg,
        trust_remote_code=True,
    )

    cfg = GRPOConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=1,
        learning_rate=1.5e-5,
        lr_scheduler_type="linear",
        warmup_ratio=0.03,
        logging_steps=10,
        save_steps=100,
        bf16=True,
        group_size=args.group_size,
        max_new_tokens=args.max_new_tokens,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_fn=build_reward_fn(args.max_new_tokens),
        args=cfg,
        train_dataset=ds,
        tokenizer=tokenizer,
        formatting_func=lambda batch: batch["prompt"],
        reward_kwargs={"answers": ds["answer"]},
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
