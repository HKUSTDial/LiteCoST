"""
Data Verification: this script is used to filter out non-cot format/think data and poor quality data.
"""
import json
import os
import re
import random
import time

import llm
import llm.global_config as config

from src.reasoner import reasoning
from src.extract.to_table import parse_table
from test import read_json, save_to_json, merge_json_files, check_answer, create_folder


# Filter out non-cot format/think data
def filter_notcot(json_file, output_file):
    """
    Filter out non-cot format/think data

    Args:
        json_file (str): Path to input JSON file
        output_file (str): Path to output JSON file
    """
    data1 = read_json(json_file)

    results = []
    for item in data1:
        id_ = item["id"]

        # Get schema and data_structure
        schema = item.get('schema', [])
        data_structure = item['data_structure']
        structure_data = item['structured_data']
        steps = item.get('steps', [])
        cot = item['cot']

        # data_structure, structured_data = structure_map.get(id_, (None, None))
        # if not data_structure: continue

        # if data_structure == "Text Description" or not schema or not steps: continue
        if data_structure == "Text Description" or not schema: continue

        if not isinstance(steps, list): 
            continue

        # Filter non-<think></think>...<answer></answer>
        pattern = r"^<reasoning>[\s\S]*?</reasoning>\s*<answer>[\s\S]*?</answer>$"
        if re.match(pattern, cot):
            results.append(item)

    print(f"After first filtering, length: {len(results)}")
    save_to_json(results, output_file)

# Filter poor quality data: data that cannot be answered
def filter_quality(structured_data_file, output_file_true, output_file_false, folder_path):
    """
    Filter out poor quality data: data that cannot be answered

    Args:
        structured_data_file (str): Path to input JSON file
        output_file_true (str): Path to output JSON file
        output_file_false (str): Path to output JSON file
        folder_path (str): Path to folder containing JSON files
    """
    create_folder(folder_path)

    if os.path.exists(structured_data_file):
        print(11)
        structured_datas = read_json(structured_data_file)

    qa_datas = []
    final_results = []
    check_result = False

    with open(qa_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                record = json.loads(line.strip())
                qa_datas.append(record)

    for record in qa_datas:
        # id = record["financebench_id"]
        # file_id = id

        id = record["id"]
        file_id = id.replace("/", "_") 
        # file_id = id # TATQA
        record_file = f"{result_data_folder}/{file_id}.json"

        if os.path.exists(record_file):
            continue

        std_ans = record['answer']
        question = record["question"]

        structured_data = next((sd for sd in structured_datas if sd["id"] == id), None)
        
        if not structured_data: continue

        structured_data = structured_data['structured_data']
        # answer = structured_data['answer'] # (table<|>..)
        # structured_data = structured_data['cot'] #cot-structure
        print(id, question)
        print(structured_data)

        final_ans, extracted_ans = reasoning(question, structured_data)
        # check_result = check_answer(question, final_ans, std_ans)

        print("-----")

        final_results.append({
            "id": id,
            "question": question,
            'answer': final_ans,
            'extracted_answer': extracted_ans,
            'std_answer': std_ans,
            # 'check_answer': check_result
        })
        save_to_json(final_results, record_file)
        final_results.clear()
    
    # exit()
    # check result
    add_check_results(folder_path)

    merge_json_files(result_data_folder, result_file)

    results = read_json(result_file)    

    true_ids = {item["id"] for item in results if item["check_answer"] == True}
    false_ids = {item["id"] for item in results if item["check_answer"] == False}
    filtered_structured_datas = [item for item in structured_datas if item["id"] in true_ids]
    others_structured_datas = [item for item in structured_datas if item["id"] in false_ids]
    
    save_to_json(filtered_structured_datas , output_file_true)
    save_to_json(others_structured_datas , output_file_false)

    print(f"After second filtering, length: {len(filtered_structured_datas )}")

def add_check_results(folder_path):
    """
    Process all JSON files in the folder, add check_result field and save again.

    Args:
        folder_path (str): Path to folder containing JSON files
    """
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)

            data = read_json(file_path)

            for record in data:
                if "check_answer" in record: continue

                check_result = check_answer(
                    # record["question"],
                    record["answer"],
                    record["std_answer"]
                )
                print(f"answer: {record['answer']}, check: {check_result}")

                # Add check_result field
                record["check_answer"] = check_result

            # Write modified data back to file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

            print(f"Processed and saved: {file_path}")

if __name__ == '__main__':
    model = "gpt-4o" #gpt, llama, deepseek
    config.set_model(model)

    qa_file = './dataset/Finqa/finqa_train.jsonl'
    structured_data_file = f'./results/Finqa/train/{model}/structured_data_results.json'
    structured_data_file_filter1 = f"./results/Finqa/train/{model}/structured_data_results_filter1.json"

    result_data_folder = f'./results/Finqa/train/{model}/ours_results_{model}'
    result_file = f"./results/Finqa/train/{model}/ours_{model}.json"

    output_file_true = f"./results/Finqa/train/{model}/structured_data_results_filter2.json"
    output_file_false = f"./results/Finqa/train/{model}/structured_data_results_filter2_false.json"

    first_filter = filter_notcot(structured_data_file, structured_data_file_filter1)
    filter_quality(structured_data_file_filter1, output_file_true, output_file_false, result_data_folder)

