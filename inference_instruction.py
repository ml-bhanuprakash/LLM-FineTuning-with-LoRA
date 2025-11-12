#!/usr/bin/env python3
"""
inference_instruction.py
Performs instruction-style inference on a TinyLlama model fine-tuned with LoRA adapters.
"""

import argparse
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def format_prompt(instruction: str, context: str = "") -> str:
    """Formats instruction and optional context for inference."""
    prompt = f"### Instruction:\n{instruction.strip()}\n### Input:\n{context.strip()}\n### Response:\n"
    return prompt


def load_model(model_path: str, base_model: str = None):
    """Loads fine-tuned instruction model (with optional LoRA)."""
    if base_model and PEFT_AVAILABLE:
        logger.info("Loading base model and applying LoRA adapter...")
        base = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
        model = PeftModel.from_pretrained(base, model_path)
    else:
        logger.info("Loading full fine-tuned model...")
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

    tokenizer = AutoTokenizer.from_pretrained(model_path if not base_model else base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate_response(model, tokenizer, instruction: str, context: str = "", max_new_tokens: int = 256):
    """Generates structured response for a given instruction."""
    prompt = format_prompt(instruction, context)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.7, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def main():
    parser = argparse.ArgumentParser(description="Run inference using instruction-tuned TinyLlama model.")
    parser.add_argument("--model_path", required=True, help="Path to fine-tuned instruction model.")
    parser.add_argument("--base_model", default=None, help="Base model (if LoRA adapter used).")
    parser.add_argument("--instruction", default="Summarize the pharmacological use of Ezetimibe.")
    parser.add_argument("--context", default="")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_path, args.base_model)
    logger.info("Running instruction-based inference...")
    output = generate_response(model, tokenizer, args.instruction, args.context, args.max_new_tokens)
    print("\n=== Model Output ===\n")
    print(output)


if __name__ == "__main__":
    main()
