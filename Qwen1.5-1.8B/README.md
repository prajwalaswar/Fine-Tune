# Qwen LoRA Fine-Tuning

This project demonstrates how to fine-tune the [Qwen1.5-1.8B](https://huggingface.co/Qwen/Qwen1.5-1.8B) model using Low-Rank Adaptation (LoRA) on a small dataset in the Guanaco format. The fine-tuned model is then used for text generation tasks.

## Features

- Fine-tune the Qwen1.5-1.8B model using LoRA for efficient training.
- Use the Hugging Face `datasets` and `transformers` libraries for dataset loading and model training.
- Tokenize and preprocess data for causal language modeling.
- Save and load the fine-tuned model for text generation tasks.

## Requirements

Install the required libraries by running:

```bash
pip install -q transformers datasets peft accelerate bitsandbytes trl

### Instruction:
Give three tips to stay focused while studying..

### Response:
1. Create a dedicated study space free from distractions.
2. Break your study sessions into manageable chunks with short breaks in between.
3. Use tools like to-do lists or timers to stay organized and track your progress.

Acknowledgments
Hugging Face Transformers
Hugging Face Datasets
PEFT (Parameter-Efficient Fine-Tuning)
License

This project is licensed under the MIT License. See the LICENSE file for details. ```
