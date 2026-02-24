"""
This script is used for GRPO (Generative Reward Prompt Optimization) training.
It implements reward-guided prompt optimization for improving model performance
on domain-specific information extraction tasks.

The script handles:
- Loading and preparing domain-specific information extraction datasets
- Setting up model and tokenizer configurations 
- Implementing GRPO training loop
- Saving model checkpoints and outputs
"""

import torch
import os
import json

from src.reward import format_reward, answer_reward
from src.reasoner import reasoning

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

from unsloth import FastLanguageModel


def prepare_dataset(file_path):
    """Load and prepare the Finance dataset for training."""
    with open(file_path, 'r', encoding='utf-8') as file:
        formatted_data = json.load(file)

    return formatted_data


max_seq_length = 120000 # Can increase for longer reasoning traces
lora_rank = 32 # Larger rank = smarter, but slower


model_path = "Llama-3.2-3B-Instruct"
lora_path = "./model/lora_model/"
outputs = "./model/grpo_model"
checkpoint_dir = os.path.join(outputs, "checkpoints")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = lora_path, # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)
# model = FastLanguageModel.get_peft_model(
#     model,
#     r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
#     target_modules = [
#         "q_proj", "k_proj", "v_proj", "o_proj",
#         "gate_proj", "up_proj", "down_proj",
#     ], # Remove QKVO if out of memory
#     lora_alpha = lora_rank,
#     use_gradient_checkpointing = "unsloth", # Enable long context finetuning
#     random_state = 3407,
# )



max_prompt_length = 110000
dataset = prepare_dataset("./dataset/finance_train_grpo.json")
# print(dataset[0])

from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "paged_adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4, # Increase to 4 for smoother training
    num_generations = 4, # Decrease if out of memory
    max_prompt_length = max_prompt_length,
    max_completion_length = 2048,
    num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 2,
    save_steps = 1,
    max_grad_norm = 0.1,
    report_to = "none", # Can use Weights & Biases
    output_dir = outputs,
    resume_from_checkpoint = checkpoint_dir if os.path.exists(checkpoint_dir) else None,
)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [format_reward, answer_reward],
    args = training_args,
    train_dataset = dataset,
)
trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)



# model.save_lora(f"{outputs}/grpo_saved_lora")
model.save_pretrained(outputs)
tokenizer.save_pretrained(outputs)
model.save_pretrained_merged(f"{outputs}/merged", tokenizer, save_method = "merged_16bit",)


