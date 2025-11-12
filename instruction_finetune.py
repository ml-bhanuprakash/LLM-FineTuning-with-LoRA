#!/usr/bin/env python3
"""
instruction_finetune.py
Production-style script for instruction fine-tuning.
- Loads an instruction-style dataset (CSV/JSONL) or HuggingFace dataset
- Formats examples into a single 'text' prompt ready for tokenization
- Fine-tunes a causal LM (optionally using LoRA)
Usage example:
    python instruction_finetune.py \
      --model_name TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
      --data_path ./pharma_instruction_data.csv \
      --output_dir ./outputs/instruct \
      --epochs 3 --batch_size 1
"""
from pathlib import Path
import argparse
import logging
from typing import Dict

import pandas as pd
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

# Optional imports for LoRA (PEFT)
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_csv_or_jsonl(path: Path):
    """Load CSV or JSONL into a Hugging Face Dataset."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
        ds = Dataset.from_pandas(df)
    elif suffix in (".json", ".jsonl", ".ndjson"):
        ds = load_dataset("json", data_files=str(path), split="train")
    else:
        raise ValueError("Unsupported file format. Provide CSV or JSONL.")
    return ds


def format_instruction_example(example: Dict) -> Dict:
    """
    Expected columns: instruction, input, output
    Produces a single text field in the form:
    ### Instruction:
    <instruction>
    ### Input:
    <input>
    ### Response:
    <output>
    """
    instruction = example.get("instruction", "") or ""
    inp = example.get("input", "") or ""
    output = example.get("output", "") or example.get("response", "") or ""
    text = f"### Instruction:\n{instruction}\n### Input:\n{inp}\n### Response:\n{output}"
    return {"text": text}


def prepare_and_tokenize(dataset, tokenizer, max_length=512):
    """Map dataset to 'text' and tokenize for causal LM."""
    # Ensure example has 'text' column
    if "text" not in dataset.column_names:
        dataset = dataset.map(format_instruction_example)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tok_fn(example):
        tokens = tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized = dataset.map(tok_fn, batched=False, remove_columns=["text"])
    return tokenized


def build_lora_if_requested(model, r=8, alpha=16, target_modules=None):
    """Wrap model with LoRA if PEFT is available. Returns model (possibly wrapped)."""
    if not PEFT_AVAILABLE:
        raise RuntimeError("PEFT not available. Install `peft` to use LoRA.")
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    return model


def train(tokenized_ds, model_name: str, output_dir: str, epochs: int, batch_size: int, lr: float, use_lora: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    if use_lora:
        logger.info("Wrapping model with LoRA (PEFT)")
        model = build_lora_if_requested(model)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=8 if batch_size == 1 else 1,
        save_total_limit=2,
        logging_steps=20,
        learning_rate=lr,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(output_dir)
    logger.info("Instruction fine-tuning finished. Model saved to %s", output_dir)
    return model, tokenizer


def generate_sample(model, tokenizer, prompt: str, max_new_tokens: int = 128):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Instruction fine-tuning (production style).")
    parser.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    parser.add_argument("--data_path", type=str, required=True, help="CSV or JSONL file with instruction dataset.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA via PEFT (if installed).")

    args = parser.parse_args()
    data_path = Path(args.data_path)

    ds = load_csv_or_jsonl(data_path)
    tokenized = prepare_and_tokenize(ds, AutoTokenizer.from_pretrained(args.model_name), max_length=args.max_length)

    model, tokenizer = train(
        tokenized_ds=tokenized,
        model_name=args.model_name,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_lora=args.use_lora,
    )

    # Sample generation
    try:
        prompt = "Explain the mechanism of action of Metformin."
        result = generate_sample(model, tokenizer, prompt, max_new_tokens=100)
        logger.info("Sample generation:\n%s", result)
    except Exception:
        logger.exception("Sample generation failed.")


if __name__ == "__main__":
    main()
