# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the CoST dataset to parquet format
"""

import argparse
import os
import re
import json

from datasets import load_dataset, Dataset

from verl.utils.hdfs_io import copy, makedirs


def extract_solution(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

# instructions format
SYSTEM_PROMPT = """
Please reason step by step, and respond in the following format: 
<reasoning>
...
</reasoning>
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


def prepare_dataset2(file_path, if_instruction=True):
    """Load and prepare the Finance dataset for training."""
    if if_instruction:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        # 解析每行并存储为列表
        dataset  = [json.loads(line.strip()) for line in lines]
        dataset = Dataset.from_list(dataset)


    else:
        with open(file_path, 'r', encoding='utf-8') as file:
            dataset = json.load(file)
        dataset = [data['messages'] for data in dataset]
        # print(dataset[0])
        if isinstance(dataset, list):
            dataset = Dataset.from_dict({"messages": dataset})
                
    return dataset

def prepare_dataset(file_path, if_instruction=True):
    """Load and prepare the Finance dataset for training."""
    with open(file_path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)
    # 处理新的数据格式
    if isinstance(dataset, list):
        processed_data = []
        for item in dataset:
            if 'prompt' in item and 'answer' in item:
                # 提取system和user消息
                system_msg = next((msg['content'] for msg in item['prompt'] if msg['role'] == 'system'), None)
                user_msg = next((msg['content'] for msg in item['prompt'] if msg['role'] == 'user'), None)
                
                # 构建标准格式
                processed_item = {
                    'messages': [
                        {'role': 'system', 'content': system_msg} if system_msg else None,
                        {'role': 'user', 'content': user_msg} if user_msg else None,
                        {'role': 'assistant', 'content': item['answer']}
                    ]
                }
                # 移除None值
                processed_item['messages'] = [msg for msg in processed_item['messages'] if msg is not None]
                processed_data.append(processed_item)
        dataset = Dataset.from_list(processed_data)
                
    return dataset

if __name__ == "__main__":
    # python -m examples.data_preprocess.cost --local_dir ./data/cost_scientificqa/chunk
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./data/cost_legal/table")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "litesea/cost_chunk"

    # train_dataset = prepare_dataset2("/data/liangzhuowen/projects/DocumentAI/grpo_results/mixed/sft.json", if_instruction=False)
    # test_dataset = prepare_dataset("/data/liangzhuowen/projects/DocumentAI/grpo_results/mixed/grpo.json")

    # train_dataset = prepare_dataset2("/data/liangzhuowen/projects/DocumentAI/grpo_results/LegalBench/table/sft.json", if_instruction=False)
    # test_dataset = prepare_dataset("/data/liangzhuowen/projects/DocumentAI/grpo_results/LegalBench/table/grpo.json")
    

    train_dataset = prepare_dataset2("/data/liangzhuowen/projects/DocumentAI/grpo_results/chunks/sft.json", if_instruction=False)
    test_dataset = prepare_dataset("/data/liangzhuowen/projects/DocumentAI/grpo_results/chunks/grpo_process.json")
    

    # print(train_dataset[0]['messages'])
    # content = train_dataset[0]['messages'][1]['content']
    # content = json.loads(content)
    # print(content['schema'])
    # exit()

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            # 获取 messages 数组
            messages = example['messages']
            
            # 提取 system、user 和 assistant 消息
            system_msg = next((msg['content'] for msg in messages if msg['role'] == 'system'), None)
            user_msg = next((msg['content'] for msg in messages if msg['role'] == 'user'), None)
            assistant_msg = next((msg['content'] for msg in messages if msg['role'] == 'assistant'), None)

            solution = extract_solution(assistant_msg)
            
            # schema = json.loads(user_msg)['schema']
            question = json.loads(user_msg)['question']

            # 构建 VERL 标准格式
            data = {
                "data_source": data_source,  # 根据您的数据集类型修改
                "prompt": [
                    {
                        "role": "system",
                        "content": system_msg if system_msg else SYSTEM_PROMPT  # 使用原始系统提示词，如果没有则使用默认的
                    },
                    {
                        "role": "user",
                        "content": user_msg
                    }
                ],
                "ability": "chunk_construction",  # 根据任务类型修改
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "schema_question": question,
                    "system_prompt": system_msg,
                    "question": user_msg,
                    "answer": assistant_msg
                }
            }
            return data
        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True,remove_columns=train_dataset.column_names) #移除原始数据集的列
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True,remove_columns=test_dataset.column_names) #移除原始数据集的列
    
    print(train_dataset[0])
    print(len(train_dataset))
    print(test_dataset[0])
    print(len(test_dataset))
    # exit()
    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))


    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
