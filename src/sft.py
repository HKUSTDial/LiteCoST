"""
Python script for fine-tuning large language models using the unsloth library.

Main functionalities:
1. Dataset preparation and formatting
2. SFT model training 
3. Model inference and testing
4. Logging and checkpoint saving
5. Evaluation and results saving
"""


import torch
import os, json, sys

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only

import logging
def get_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    r"""
    Get a standard logger that outputs logs to both console and a file at the specified path.
    """
    # 确保日志文件所在的目录存在
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_txt = os.path.join(log_dir, "log.txt")  

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S"
    )
    # 创建 StreamHandler（输出到控制台）
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    # 创建 FileHandler（输出到 log.txt）
    file_handler = logging.FileHandler(log_txt, mode="w")  # 'w' 代表覆盖写入，'a' 代表追加
    file_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(stream_handler)  # 添加到控制台
    logger.addHandler(file_handler)  # 添加到文件

    return logger


max_seq_length = 110000 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 2x faster
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # 4bit for 405b!
    "unsloth/Mistral-Small-Instruct-2409",     # Mistral 22b 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!

    "unsloth/Llama-3.2-1B-bnb-4bit",           # NEW! Llama 3.2 models
    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "unsloth/Llama-3.2-3B-bnb-4bit",
    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",

    "unsloth/Llama-3.3-70B-Instruct-bnb-4bit" # NEW! Llama 3.3 70B!
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "model_name", # or choose "unsloth/Llama-3.2-1B-Instruct"
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    device_map = 'auto',
    cache_dir = "model_path"
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)

# gpt format
def formatting_prompts_func_gpt(examples):
    """
    Format the prompts for the GPT model.
    """
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }
pass

# instructions format
SYSTEM_PROMPT = """
Please reason step by step, and respond in the following format: 
<think>
...
</think>
<answer>
...
</answer>
"""

instruction_prompt = f"""{SYSTEM_PROMPT}
### Instruction:
{{}}
### Input:
{{}}
### Response:
{{}}"""

EOS_TOKEN = tokenizer.eos_token # Add EOS_TOKEN
def formatting_prompts_func_ins(examples):
    """
    Format the prompts for the instruction model.
    """
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Add EOS_TOKEN, otherwise infinite generation
        text = instruction_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

from datasets import load_dataset, Dataset
from unsloth.chat_templates import standardize_sharegpt


def prepare_dataset(file_path, if_instruction=True):
    """Load and prepare the Finance dataset for training."""
    if if_instruction:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        # 解析每行并存储为列表
        dataset  = [json.loads(line.strip()) for line in lines]
        dataset = Dataset.from_list(dataset)

        # print(dataset[0])
        dataset = dataset.map(formatting_prompts_func_ins, batched = True,)

    else:
        with open(file_path, 'r', encoding='utf-8') as file:
            dataset = json.load(file)
        dataset = [data['messages'] for data in dataset]
        # print(dataset[0])
        if isinstance(dataset, list):
            dataset = Dataset.from_dict({"messages": dataset})
        print(dataset)
        # dataset = standardize_sharegpt(dataset) # 将数据集格式从ShareGPT格式转换为Hugging Face的标准化结构
        dataset = dataset.map(formatting_prompts_func_gpt, batched = True,)
        
    return dataset




lora_path = "lora_path"
def main():
    """
    Main function to train the model.
    """
    dataset = prepare_dataset("data_path", if_instruction=False)
    print(dataset[0]['text'])

    # 初始化 logger
    logger = get_logger("TrainingLogger", log_dir=lora_path)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            num_train_epochs = 3, # Set this for 1 full training run.
            # max_steps = 1000,
            # save_steps = 500,
            save_strategy = "epoch", 
            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 10,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = lora_path,
            report_to = "none", # Use this for WandB etc
        ),
    )

    # gpt
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
        response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
    )


    # 训练开始前记录日志
    logger.info("Starting the training process...")

    # 开始训练并记录训练状态
    trainer_stats = trainer.train()

    # 训练完成后记录日志
    logger.info(f"Training completed. Final stats: {trainer_stats}")

    model.save_pretrained(lora_path)  # Local saving
    tokenizer.save_pretrained(lora_path)
    model.save_pretrained_merged(f"{lora_path}/merged", tokenizer, save_method = "merged_16bit",)


def get_answer_instruction(context, schema, model, tokenizer):
    """
    Get the answer from the instruction model.
    """
    prompt = f'''
    {{
    "instruction": "You are an expert in table construction. Please extract entities that match the schema definition from the input, and finally generate the structured table.",
    "schema": {schema}
    }}
    '''
    inputs = tokenizer(
    [
        instruction_prompt.format(
            prompt, # instruction
            context, # input
            "", # output - leave this blank for generation!
        )
    ], return_tensors = "pt").to("cuda")
    from transformers import TextStreamer
    text_streamer = TextStreamer(tokenizer)
    test_response = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 4096)
    
    print(test_response)
    
    # if "</answer>" in test_response[0]:
        # test_response = test_response[0].split("</answer>")[0] + "</answer>"
    
    return test_response


def get_answer_gpt(context, schema, model, tokenizer):
    """
    Get the answer from the GPT model.
    """
    prompt = f'''
    {{
    "instruction": "You are an expert in table construction. Please extract entities that match the schema definition from the input, and finally generate the structured table.",
    "schema": {schema},
    "input": "{context}"
    }}
    '''

    test_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    # print(test_messages)
    inputs = tokenizer.apply_chat_template(
        test_messages,
        tokenize = True,
        add_generation_prompt = True, # Must add for generation
        return_tensors = "pt",
    ).to("cuda")
    input_length = inputs.shape[1]

    outputs = model.generate(input_ids = inputs, max_new_tokens = 2048, use_cache = True,
                        temperature = 1.5, min_p = 0.1)

    test_response = tokenizer.batch_decode(outputs[:, input_length:], skip_special_tokens=True)

    if "</answer>" in test_response[0]:
        test_response = test_response[0].split("</answer>")[0] + "</answer>"
    
    return test_response

def test(test_file, save_file):
    """
    Test the model.
    """
    print(111)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = lora_path, # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    FastLanguageModel.for_inference(model)

    preds = []
    labels = []
    with open(test_file, "r", encoding="utf-8") as f:
        datas = json.load(f)
    prompts_to_test = [data['messages'][1]['content'] for data in datas]
    labels = [data['messages'][2]['content'] for data in datas]

    for prompt in prompts_to_test:
        # Prepare the prompt using the chat format supported by the Qwen model.
        test_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        # print(test_messages)
        inputs = tokenizer.apply_chat_template(
            test_messages,
            tokenize = True,
            add_generation_prompt = True, # Must add for generation
            return_tensors = "pt",
        ).to("cuda")
        input_length = inputs.shape[1]

        outputs = model.generate(input_ids = inputs, max_new_tokens = 4096, use_cache = True,
                            temperature = 1.5, min_p = 0.1)

        test_response = tokenizer.batch_decode(outputs[:, input_length:], skip_special_tokens=True)

        if "</answer>" in test_response[0]:
            test_response = test_response[0].split("</answer>")[0] + "</answer>"
        else:
            test_response = test_response[0]
        print("\nModel Response:")
        print(test_response)

        preds.append(test_response) 
        # break

    with open(save_file, "w", encoding="utf-8") as f:
        for text, pred, label in zip(prompts_to_test, preds, labels):
            f.write(json.dumps({"instruction": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")




if __name__ == "__main__":
    main()
    