# Llama-2-7B Fine-Tuning with QLoRA

This repository contains code to fine-tune the **Llama-2-7B Chat** model using **QLoRA** and the **Guanaco Dataset** for instruction-tuned tasks. The process uses efficient techniques like 4-bit quantization and LoRA (Low-Rank Adaptation) to reduce memory requirements while maintaining performance.

---

## 🚀 Features
- **Model:** Fine-tuned [NousResearch/Llama-2-7b-chat-hf](https://huggingface.co/NousResearch/Llama-2-7b-chat-hf).
- **Dataset:** [mlabonne/guanaco-llama2-1k](https://huggingface.co/datasets/mlabonne/guanaco-llama2-1k).
- **Fine-Tuning:** Applied QLoRA with configurable parameters for:
  - LoRA dimensions, scaling, and dropout.
  - 4-bit precision for memory efficiency.
- **Training Framework:** Hugging Face Transformers + TRL (Transformers Reinforcement Learning).

---

## 📦 Dependencies
- `transformers==4.31.0`
- `accelerate==0.21.0`
- `peft==0.4.0`
- `bitsandbytes==0.40.2`
- `trl==0.4.7`

Install all dependencies with:
```bash
pip install -r requirements.txt
