# 🧠 Fine-Tune Qwen 1.5 (1.8B) with QLoRA on Google Colab

This project shows how to fine-tune **Alibaba’s Qwen 1.5 (1.8B)** model using **QLoRA (Parameter-Efficient Fine-Tuning)** on the Hugging Face dataset `timdettmers/openassistant-guanaco`. The notebook is designed to run on **Google Colab Free Tier** with a GPU.

---

## 🚀 Features

- ✅ Load Qwen 1.5 (1.8B) model and tokenizer
- ✅ Use PEFT (QLoRA) for efficient training
- ✅ Use small HF dataset to stay within Colab limits
- ✅ Save fine-tuned model
- ✅ Perform inference inside the notebook

---

## 🛠️ Tech Stack

- 🤗 Hugging Face Transformers
- 🤗 PEFT (QLoRA)
- 🤗 Datasets
- 🤗 Accelerate
- Google Colab (Free GPU)

---

## 📂 Dataset

- **Name:** [`timdettmers/openassistant-guanaco`](https://huggingface.co/datasets/timdettmers/openassistant-guanaco)
- **Type:** Instruction → Response format
- **Purpose:** Train model to follow human-like instructions

---

## 📋 Setup Instructions

### 1. Open Google Colab
- Go to: [https://colab.research.google.com](https://colab.research.google.com)
- Make sure GPU is enabled (`Runtime` → `Change runtime type` → select `GPU`)

### 2. Run Notebook Step-by-Step
- Install libraries
- Load model and dataset
- Apply QLoRA using `peft`
- Train model
- Run inference
- Save the fine-tuned model

---

## 🧪 Sample Inference

```python
prompt = "### Instruction:\nWhat is QLoRA?\n\n### Response:\n"
output = pipe(prompt, max_new_tokens=100, temperature=0.7)
print(output[0]["generated_text"])
