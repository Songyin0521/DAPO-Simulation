# GRPO-Math (Reproduction Template - v2)

This repo reproduces a math reasoning GRPO training on GSM8K with improved tricks:
Token-Level PG, GRPO group advantage, ClipHigher, Overlong penalty, Dynamic Scaled Gradient, QLoRA.

## Quickstart (Windows PowerShell)

```powershell
# 1) Create venv
python -m venv .venv
.\.venv\Scripts\activate

# 2) Install deps
pip install -U pip
pip install -r requirements.txt
# Install PyTorch matching CUDA (choose one)
pip install torch --index-url https://download.pytorch.org/whl/cu121
# or: pip install torch --index-url https://download.pytorch.org/whl/cu118

# 3) Prepare data
python src/prepare_gsm8k.py

# 4) Train
python src/train_grpo.py --model_name Qwen/Qwen2.5-1.5B-Instruct --math_variant true

# 5) Evaluate
python src/evaluate.py --ckpt_path outputs/last --k 8 --n_eval 200
```
