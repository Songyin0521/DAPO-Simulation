import argparse, pathlib, math, torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from rewards import reward_math, normalize_group_advantages
from dataset import load_jsonl, batchify_indices
from utils import set_seed, EMAVar, setup_bnb_4bit
from sampling import generate_k

def parse_args():
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--model_name", type=str, default=None)
    ap.add_argument("--math_variant", action="store_true")
    return ap.parse_args(), yaml

def main():
    args, yaml = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.model_name: cfg["model_name"] = args.model_name
    if args.math_variant: cfg["math_variant"] = True

    set_seed(cfg["seed"])
    outdir = pathlib.Path(cfg["output_dir"]); outdir.mkdir(parents=True, exist_ok=True)

    run_name = f"{cfg['model_name'].split('/')[-1]}-grpo"
    wandb.init(project="grpo-math", name=run_name, config=cfg)

    train = load_jsonl("data/gsm8k_train.jsonl")

    quant_config = setup_bnb_4bit(bfloat16=cfg.get("bf16", True)) if cfg.get("bnb_4bit", True) else None
    tok = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        device_map="auto",
        quantization_config=quant_config,
        trust_remote_code=True
    )
    if quant_config:
        model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_cfg)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = get_linear_schedule_with_warmup(opt, cfg["warmup_steps"], cfg["max_steps"])
    scaler_var = EMAVar(cfg["dsd_tau"])

    model.train()
    grad_accum = max(1, int(cfg.get("gradient_accumulation_steps", 1)))
    update_step = 0
    micro_step = 0
    gsize = cfg["group_size"]
    done = False

    while not done:
        for idx in batchify_indices(len(train), cfg["per_device_train_batch_size"]):
            prompts = [train.rows[i]["prompt"] for i in idx]
            golds   = [train.rows[i]["answer"] for i in idx]

            texts_group, hitmax_group = generate_k(model, tok, prompts, k=gsize, max_new_tokens=cfg["max_new_tokens"], temperature=0.7)

            batch_rewards = []
            for texts, hitmax, gold in zip(texts_group, hitmax_group, golds):
                rewards = reward_math(texts, [gold]*len(texts), hitmax,
                                      overlong_penalty=cfg["overlong_penalty"],
                                      length_penalty_alpha=cfg["length_penalty_alpha"],
                                      max_new_tokens=cfg["max_new_tokens"],
                                      tokenizer=tok)
                batch_rewards.append(rewards)

            all_loss = 0.0
            for b in range(len(prompts)):
                # NLL surrogate for token-level PG (efficient approximation)
                cont_inputs = tok(texts_group[b], return_tensors="pt", padding=True, truncation=True, max_length=cfg["max_prompt_len"]+cfg["max_new_tokens"]).to(model.device)
                labels = cont_inputs["input_ids"].clone()
                out = model(**cont_inputs, labels=labels)
                # keep grad for backward; do not detach
                nll = out.loss * torch.ones(len(texts_group[b]), device=model.device)

                import torch as T
                adv = T.tensor(normalize_group_advantages(batch_rewards[b]), device=model.device)
                if cfg["cliphigher"] and cfg["cliphigher"]>0:
                    adv = T.where(adv > cfg["cliphigher"], T.tensor(cfg["cliphigher"], device=adv.device), adv)

                var = scaler_var.update(adv)
                scale = float(min(max((1.0 / (var**0.5)), cfg["dsd_min"]), cfg["dsd_max"]))
                loss_group = (nll * (-adv)).mean() * scale
                all_loss = all_loss + loss_group

            all_loss = all_loss / max(1, len(prompts))
            # scale loss for gradient accumulation
            (all_loss / grad_accum).backward()
            micro_step += 1

            if micro_step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); scheduler.step(); opt.zero_grad()
                update_step += 1

                if update_step % cfg["log_every"] == 0:
                    total_rewards = sum(sum(r) for r in batch_rewards)
                    total_counts = sum(len(r) for r in batch_rewards)
                    avg_reward = float(total_rewards / max(1, total_counts))
                    print(f"step {update_step} | loss {all_loss.item():.4f} | reward {avg_reward:.3f} | scale={scale:.3f}")
                    wandb.log({"step": update_step, "loss": all_loss.item(), "avg_reward": avg_reward, "scale": scale})

                if update_step % cfg["save_every"] == 0:
                    ck = outdir / "last"
                    ck.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(ck); tok.save_pretrained(ck)

                if update_step >= cfg["max_steps"]:
                    done = True
                    break

            if done:
                break

    print("Training done. Checkpoints at outputs/last")
    wandb.finish()

if __name__ == "__main__":
    main()
