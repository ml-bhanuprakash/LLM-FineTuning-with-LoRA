#!/usr/bin/env python3
"""
non_instruction_pretrain_finetune.py
Production-style script for non-instruction (causal LM) fine-tuning.
- Loads plain text datasets or extracts text from PDFs
- Tokenizes and prepares examples for causal LM training
- Trains using Hugging Face Trainer (optionally with LoRA via PEFT)
Usage example:
    python non_instruction_pretrain_finetune.py \
      --model_name TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
      --pdf_dir ./pdfs \
      --output_dir ./outputs/noninstr \
      --epochs 2 --batch_size 2
"""
from pathlib import Path
import argparse
import logging
import re
from typing import List, Dict

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

# Optional imports for LoRA (PEFT). Import only if available.
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def extract_text_from_pdf(pdf_path: Path) -> List[str]:
    """Extract text blocks from a single PDF file (requires PyMuPDF/fitz installed)."""
    try:
        import fitz  # PyMuPDF
    except Exception as e:
        logger.error("fitz (PyMuPDF) is required to parse PDFs. Install via `pip install PyMuPDF`.")
        raise e

    texts = []
    with fitz.open(str(pdf_path)) as doc:
        for page in doc:
            text = page.get_text("text").strip()
            if text:
                texts.append(text)
    return texts


def split_paragraphs(pages: List[str], min_len: int = 30) -> List[Dict]:
    """Split page texts into paragraphs/chunks suitable for LM training."""
    paragraphs = []
    for page_text in pages:
        # split on double new-line or long newline runs
        chunks = re.split(r'\n\s*\n', page_text)
        for chunk in chunks:
            clean = chunk.strip()
            if len(clean) >= min_len:
                paragraphs.append({"text": clean})
    return paragraphs


def load_texts_from_dir(pdf_dir: Path) -> List[Dict]:
    """Given a directory, extract text from all PDFs and return list of dict examples."""
    examples = []
    if not pdf_dir.exists():
        logger.warning("pdf_dir does not exist: %s", pdf_dir)
        return examples

    for p in sorted(pdf_dir.glob("*.pdf")):
        logger.info("Extracting text from %s", p)
        try:
            pages = extract_text_from_pdf(p)
            paragraphs = split_paragraphs(pages)
            examples.extend(paragraphs)
        except Exception as exc:
            logger.exception("Failed to extract %s: %s", p, exc)
    return examples


def prepare_dataset_from_list(text_list: List[Dict]) -> Dataset:
    """Create a Hugging Face Dataset from a list of {'text': ...} dicts."""
    if not text_list:
        raise ValueError("No text examples provided.")
    ds = Dataset.from_list(text_list)
    return ds


def tokenize_dataset(dataset: Dataset, tokenizer, max_length: int = 512) -> Dataset:
    """Tokenize dataset for causal LM training. Returns dataset with labels=input_ids."""
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
        save_total_limit=2,
        logging_steps=50,
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
    logger.info("Training finished. Model saved to %s", output_dir)
    return model, tokenizer


def generate_sample(model, tokenizer, prompt: str, max_new_tokens: int = 64):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.8)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Non-instruction fine-tuning (production style).")
    parser.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    parser.add_argument("--pdf_dir", type=str, default=None, help="Directory containing PDFs to use as training text.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA via PEFT (if installed).")

    args = parser.parse_args()

    # Load data
    examples = []
    if args.pdf_dir:
        examples = load_texts_from_dir(Path(args.pdf_dir))

    if not examples:
        logger.error("No training text found. Provide --pdf_dir with PDFs or modify script to load other sources.")
        return

    ds = prepare_dataset_from_list(examples)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized = tokenize_dataset(ds, tokenizer, max_length=args.max_length)
    model, tokenizer = train(
        tokenized_ds=tokenized,
        model_name=args.model_name,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_lora=args.use_lora,
    )

    # Example generation
    sample_prompt = "Clinical trials demonstrated that combining Atorvastatin with Ezetimibe"
    try:
        text = generate_sample(model, tokenizer, sample_prompt, max_new_tokens=100)
        logger.info("Sample generation:\n%s", text)
    except Exception:
        logger.exception("Generation failed (device/formatting issue).")


if __name__ == "__main__":
    main()
