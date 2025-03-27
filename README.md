# 🧠 Fine-Tuning Phi-3 Mini for Medical Instruction Dataset

This project demonstrates how to fine-tune the `microsoft/Phi-3-mini-4k-instruct` language model on a custom medical instruction dataset using the Hugging Face Transformers and PEFT (Parameter-Efficient Fine-Tuning) libraries.

## 📘 Project Overview

- **Model**: [`microsoft/Phi-3-mini-4k-instruct`](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
- **Dataset**: Medical instruction dataset hosted on Hugging Face Hub
- **Objective**: Fine-tune an LLM to respond to domain-specific instruction-response pairs from the medical domain
- **Method**: Used LoRA (Low-Rank Adaptation) via the `peft` library for efficient fine-tuning
- **Tokenizer**: Chat-style formatting using `apply_chat_template`

## 🛠️ Key Components

### 🔹 Data Preprocessing
- Loaded dataset from: `hf://datasets/Shekswess/medical_gemma_instruct_dataset`
- Applied Hugging Face chat template formatting to convert instructions into prompt-response format
- Split into training and testing datasets using `train_test_split`

### 🔹 Model & Training
- Initialized base model and tokenizer from Microsoft’s Phi-3 mini
- Configured LoRA layers targeting `o_proj` and `qkv_proj`
- Converted DataFrame to Hugging Face `Dataset` format
- Configured training parameters using `TrainingArguments` and trained with `SFTTrainer` from `trl`

### 🔹 Checkpoint & Inference
- Model checkpoints saved at: `./model_check/checkpoint-*`
- Used saved checkpoint to reload model and tokenizer for inference

## 🧪 Tech Stack

- Python
- Hugging Face Transformers
- Hugging Face Datasets
- `peft` for LoRA
- `trl` for SFTTrainer
- Scikit-learn (for splitting)
- PyTorch
- Google Colab or Jupyter Notebook

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install transformers peft trl datasets scikit-learn
