#!/usr/bin/env python3
"""
inference_non_instruction.py
Performs text generation using the domain-adapted (non-instruction) TinyLlama model.
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


def load_model(model_path: str, base_model: str = None):
    """
    Loads a fine-tuned model. If LoRA adapters are present, merges them into the base model.
    """
    if base_model and PEFT_AVAILABLE:
        logger.info("Loading base model with LoRA adapter...")
        base = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
        model = PeftModel.from_pretrained(base, model_path)
    else:
        logger.info("Loading standard fine-tuned model...")
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

    tokenizer = AutoTokenizer.from_pretrained(model_path if not base_model else base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 150, temperature: float = 0.7):
    """Generate response for a given prompt."""
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
    outputs = pipe(prompt, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=True)
    return outputs[0]["generated_text"]


def main():
    parser = argparse.ArgumentParser(description="Run inference on non-instruction fine-tuned model.")
    parser.add_argument("--model_path", required=True, help="Path to fine-tuned model directory.")
    parser.add_argument("--base_model", default=None, help="Base model (if LoRA adapters used).")
    parser.add_argument("--prompt", default="Describe the mechanism of action of Atorvastatin.")
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_path, args.base_model)
    logger.info("Generating output...")
    text = generate_text(model, tokenizer, args.prompt, args.max_new_tokens, args.temperature)
    print("\n=== Generated Text ===\n")
    print(text)


if __name__ == "__main__":
    main()
