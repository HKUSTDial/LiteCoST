import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

"""
CUDA_VISIBLE_DEVICES=4,5 python scripts/merge_lora_adapter.py \
    --base_model_path /data/liangzhuowen/hf_models/Llama-3.2-3B-Instruct \
    --lora_adapter_path checkpoints/cost-sft/scientificqa/chunk/llama3.2-3b-ins/global_step_789 \
    --output_dir merged/scientificqa/cost_chunk-sft/llama3.2-3b-ins
"""
def main():
    parser = argparse.ArgumentParser(description="Merge a PEFT LoRA adapter into a base model.")
    parser.add_argument(
        "--base_model_path",    
        type=str,
        required=True,
        help="Path to the base model (e.g., /path/to/Llama-3.2-3B-Instruct).",
    )
    parser.add_argument(
        "--lora_adapter_path",
        type=str,
        required=True,
        help="Path to the trained LoRA adapter checkpoint (e.g., /path/to/global_step_1566).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the merged model.",
    )
    args = parser.parse_args()

    print(f"Loading base model from {args.base_model_path}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"Loading LoRA adapter from {args.lora_adapter_path}...")
    peft_model = PeftModel.from_pretrained(
        base_model,
        args.lora_adapter_path,
    )

    print("Merging the adapter into the base model...")
    merged_model = peft_model.merge_and_unload()

    print(f"Saving the merged model to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    merged_model.save_pretrained(args.output_dir)

    print("Loading and saving the tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.lora_adapter_path)
    tokenizer.save_pretrained(args.output_dir)

    print("Merge complete!")


if __name__ == "__main__":
    main() 