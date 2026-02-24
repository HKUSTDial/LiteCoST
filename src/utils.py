"""
Utility functions for the LiteSEA project.
"""
import re, os
import json


from ..llm import llm

import tiktoken
import random
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

random.seed(10086)


def token_length(text):
    return len(encoding.encode(text, disallowed_special=()))

def extract_answer_content(text):
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text

def extract_intermediate_results(text):
    pattern = r'(Step \d+:.*?)(?=Step \d+:|\Z)'
    matches = re.findall(pattern, text, re.DOTALL)
    
    return matches

def extract_intermediate_results2(text):
    # 1. Try to match the "Step X:" structure
    pattern = r'(Step \d+:.*?)(?=Step \d+:|\Z)'
    matches = re.findall(pattern, text, re.DOTALL)
    
    # 2. If "Step X:" is not found, try to match the "**标题**" structure
    if not matches:
        pattern = r'(\*\*.*?\*\*:.*?)(?=\*\*.*?\*\*:|\Z)'
        matches = re.findall(pattern, text, re.DOTALL)

    # 3. If still not found, try to match the "1. ", "2. " structure
    if not matches:
        pattern = r'(\d+\.\s+.*?)(?=\d+\.\s+|\Z)'
        matches = re.findall(pattern, text, re.DOTALL)
    
    return matches

def token_length(text):
    return len(encoding.encode(text, disallowed_special=()))

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        return data
    
def read_json_by_id(id: str, directory: str) -> dict:
    file_path = os.path.join(directory, f"{id}.json")
    
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    else:
        return f"File with id {id} does not exist in the directory."
    
def save_to_json(data_dict, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=4)

def save_to_jsonl(data, jsonl_file):
    try:
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for entry in data:
                json.dump(entry, f, ensure_ascii=False)
                f.write("\n")  # Write each entry on a new line
    except Exception as e:
        print(f"Error saving JSONL file: {e}")

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def merge_json_files(folder_path: str, output_file: str) -> None:
    merged_data = []

    # Traverse through all files in the folder
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.json'):  # Check if the file is a JSON file
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as json_file:
                        data = json.load(json_file)  # Load the JSON data
                        if isinstance(data, list):
                            merged_data.extend(data)  # Merge list data
                        else:
                            merged_data.append(data)  # Append non-list data
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    # Write the merged data to the output file
    try:
        with open(output_file, 'w', encoding='utf-8') as output_json:
            json.dump(merged_data, output_json, ensure_ascii=False, indent=4)
        print(f"Merged JSON written to {output_file}")
    except Exception as e:
        print(f"Error writing to {output_file}: {e}")


def filter_json(file_path, output_path, choice):
    """Read the JSON file and filter out the objects with check_answer=False or data_structure=text descriptions or need_recheck=False (used for grpo)"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)  # 读取 JSON 文件
    if choice == 1:
        filtered_data = [record for record in data if record.get("check_answer") is False]
    elif choice == 2:
        filtered_data = [record for record in data if record['data_structure'] != 'Text Description']
    else:
        filtered_data = [record for record in data if record['need_recheck'] == False]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, indent=4, ensure_ascii=False)

    return filtered_data

# Transform the our data to the format of Loong data
def process_loong(file_path, model):
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            count = 0  
            for line in f:
                if line.strip():  # Skip empty lines
                    try:
                        content = json.loads(line.strip())
                        id = content['id']

                        # Retrieve 'generated_ans' from the corresponding JSON file
                        if id:
                            json_file_path = f"../results/Loong/{model}/ours_results_{model}/{id}.json"  # Modify path as needed
                            if os.path.exists(json_file_path):
                                with open(json_file_path, 'r', encoding='utf-8') as ans_file:
                                    ans_data = json.load(ans_file)
                                    generated_ans = ans_data[0].get('answer', 'N/A')
                                    content['generate_response'] = generated_ans

                                data.append(content)
                                count += 1  # Increment the count of processed lines
                            else:
                                content['generate_response'] = 'N/A'

                        # print(f"Generated Answer: {content['generate_response']}")
                        # break
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line: {e}")
            print(f"Processed {count} lines")
            print(data[0].keys())  # Print the keys of the first item for reference

            # Save the processed data to a new JSONL file
            save_to_jsonl(data, f"../results/Loong/{model}/loong_generate.jsonl")
    except Exception as e:
        print(f"Error reading JSONL file: {e}")


def check_answer(ans, std_ans):
    check_ans = ""
    if str(std_ans) in ans:
        return True
    else:
        prompt = f"""
            Given a question, a standard answer, and a provided answer, your task is to verify if the provided answer is correct according to the standard answer.
            You should base your evaluation strictly on the content and correctness of the provided answer in relation to the standard answer.

            Here are the details:

            <standard answer> 
            {std_ans} 
            <standard answer>

            <provided answer> 
            {ans} 
            <provided answer> 

            Important Notes:
            - If the <provided answer> is "I don't know", "I can't get the answer", or any similar phrase, consider the answer as incorrect. 
            - If the <provided answer> is vague, irrelevant, or does not answer the question properly, return "False".
            - If the <standard answer> is a numerical value, and the format of the <provided answer> is different from that of the <standard answer> , but the numerical values are the same, then it is considered that the meanings are consistent. For example, if the <standard answer> is 0.98 and the <provided answer> is 98%, it is considered that the meanings are consistent, return "True".
            - If the <standard answer> is a numerical value, and the final result of the <provided answer> is consistent with the <standard answer> after rounding, then it is considered that the meanings are consistent. For example, if the <standard answer> is 2 and the <provided answer> is 1.98, it is considered that the meanings are consistent, return "True".

            Please respond with the result in the following format:
            {{check: True or False}}

            Do not provide any additional explanation or context, just return the result in the specified format.
        """
        check = llm.get_answer(prompt, model='gpt-4o')
        match = re.match(r'\{check:\s*([a-zA-Z\s]+)\}', check)
        if match:
            check_ans = match.group(1).strip()
        if "true" in check_ans.lower():
            return True
        else:
            return False