import json
import re


def extract_intermediate_results(text):
    """
    Extract intermediate results from the given text.

    :param text: The input text to extract results from.
    :return: A list of extracted intermediate results.
    """
    # 1. Try to match "Step X:" structure
    pattern = r'(Step \d+:.*?)(?=Step \d+:|\Z)'
    matches = re.findall(pattern, text, re.DOTALL)

    # 2. If "Step X:" is not found, try to match "**Title**" structure
    if not matches:
        pattern = r'(\*\*.*?\*\*:.*?)(?=\*\*.*?\*\*:|\Z)'
        matches = re.findall(pattern, text, re.DOTALL)

    # 3. If still not found, try to match "1. ", "2. " structure
    if not matches:
        pattern = r'(\d+\.\s+.*?)(?=\d+\.\s+|\Z)'
        matches = re.findall(pattern, text, re.DOTALL)

    return matches

def update_steps_from_cot(file_path, output_path=None):
    """
    Extract steps from COT (Chain of Thought) and update the 'steps' field in the data.

    :param file_path: Path to the input JSON file.
    :param output_path: Path to save the updated JSON file. If None, the input file will be overwritten.
    :return: The updated data.
    """
    # Read the JSON file
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Update the 'steps' field for each item
    for item in data:
        cot = item.get("cot", "")
        if cot:
            # Extract steps using extract_intermediate_results
            extracted_steps = extract_intermediate_results(cot)
            # Update the 'steps' field
            item["steps"] = extracted_steps

    # Determine the output path
    if output_path is None:
        output_path = file_path

    # Write the updated data back to the JSON file
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)  # Keep formatted storage

    print(f"Updated 'steps' field for {len(data)} records and saved to {output_path}")
    return data

def process_steps_table(data):
    """
    Process steps for data with a table structure.

    :param data: A dictionary containing the data to process.
    :return: A tuple containing the reasoning and the extracted step 5.
    """
    # Extract information from COT
    cot = data.get("cot", "")
    steps = data.get("steps", [])
    structured_data = data.get("structured_data", "")

    # 提取前三个steps
    # steps = steps[:3]  # 获取前三个步骤，去掉Step的标题部分
    # steps = cot.split("Step")[1:4]

    # reasoning = "<reasoning>\n" + "\n### Step" + "### Step".join(steps) + "\n</reasoning>\n"
    reasoning = "<reasoning>\n" + "".join(steps) + "\n</reasoning>\n"
    
    # 提取step5的部分
    step5_start = cot.rfind('(\"table\"<|>')  # 找到step5开始的位置
    step5 = cot[step5_start:]
    answer = "<answer>\n" + step5 + "\n</answer>"

    # Combine results
    result = reasoning + answer
    return result, str(step5)

def process_steps_chunks(data):
    """
    Process steps for data with a chunk structure.

    :param data: A dictionary containing the data to process.
    :return: A tuple containing the reasoning and the extracted step 5.
    """
    # Extract information from COT
    cot = data.get("cot", "")
    steps = data.get("steps", [])
    structured_data = data.get("structured_data", "")

    # 提取前三个steps
    # steps = steps[:3]  # 获取前三个步骤，去掉Step的标题部分
    # steps = cot.split("Step")[1:4]

    # reasoning = "<reasoning>\n" + "\n### Step" + "### Step".join(steps) + "\n</reasoning>\n"
    reasoning = "<reasoning>\n" + "".join(steps) + "\n</reasoning>\n"
    
    # 提取step5的部分
    # step5_start = cot.rfind('(\"chunk\"<|>')  # 找到step5开始的位置
    step5_start = -1
    # Try to match multiple possible formats
    patterns = [
        '(\"chunk\"<|>',      # Format 1: ("chunk"<|>
        '(\"chunks\"<|>',     # Format 2: ("chunks"<|>
        '\"chunk\"<|>',       # Format 3: "chunk"<|>
        '\"chunk<|>',         # Format 4: "chunk<|>
        '(\"chunk<|>',        # Format 5: ("chunk<|>
        '(\"chunk|<',
    ]

    for pattern in patterns:
        step5_start = cot.rfind(pattern)
        if step5_start != -1:
            break
    print(step5_start)
    step5 = cot[step5_start:]
    answer = "<answer>\n" + step5 + "\n</answer>"

    # Combine results
    result = reasoning + answer
    return result, str(step5)

def process_steps_graph(data):
    """
    Process steps for data with a graph structure.

    :param data: A dictionary containing the data to process.
    :return: A tuple containing the reasoning and the extracted structured data.
    """
    # Extract information from COT
    cot = data.get("cot", "")
    steps = data.get("steps", [])
    structured_data = data.get("structured_data", [])

    # 提取前三个steps
    # steps = steps[:3]  # 获取前三个步骤，去掉Step的标题部分
    # print(steps)
    # reasoning = "<reasoning>\n" + "\n### Step" + "### Step".join(steps) + "\n</reasoning>\n"
    reasoning = "<reasoning>\n" + "".join(steps) + "\n</reasoning>\n"

    # Extract step 5
    match = re.search(r'\(\"entity\".*?<\|COMPLETE\|\>\n', cot.split("Step")[-1], re.DOTALL)

    if match:
        answer = match.group(0)  # Return the matched part
    else:
        answer = "" 
    final_answer = "<answer>\n" + answer + "\n</answer>"
    final_answer = "<answer>\n" + str(structured_data) + "\n</answer>"

    # Combine results
    result = reasoning + final_answer
    return result, str(structured_data)

def process_json_file(file1, file2):
    """
    Process a JSON file and update its content based on the data structure.

    :param file1: Path to the input JSON file.
    :param file2: Path to save the processed JSON file.
    """
    # Read the JSON file
    with open(file1, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for item in data:
        if item["data_structure"] == "Table":
            item["full_cot"], item['answer'] = process_steps_table(item)
        if item["data_structure"] == "Graph":
            item["full_cot"], item['answer'] = process_steps_graph(item)
        if item["data_structure"] == "Text Description":
            item["full_cot"], item['answer'] = process_steps_chunks(item)

    # Write the updated data back to the JSON file
    with open(file2, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)  # Keep formatted storage

def filter_json(file_path, output_path, choice):
    """
    Filter a JSON file based on specific criteria.

    :param file_path: Path to the input JSON file.
    :param output_path: Path to save the filtered JSON file.
    :param choice: Filtering criteria (1, 2, or 3).
    :return: The filtered data.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)  # Read the JSON file
    if choice == 1:
        filtered_data = [record for record in data if record.get("check_answer") is False]
    elif choice == 2:
        filtered_data = [record for record in data if record['data_structure'] != 'Text description']
    else:
        filtered_data = [record for record in data if record['need_recheck'] == False]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, indent=4, ensure_ascii=False)

    return filtered_data


# 使用示例：从cot中提取steps并更新文件
# update_steps_from_cot(file1, file2)  # 更新到新文件


# 假设文件路径为'file.json'
file1 = '/data/liangzhuowen/projects/DocumentAI/results_prompt1/squad/gpt-4o_cot2/structured_data_results.json'
file2 = '/data/liangzhuowen/projects/DocumentAI/results_prompt1/squad/gpt-4o_cot2/squad.json'
# file1 = '/data/liangzhuowen/projects/DocumentAI/results_prompt1/LegalBench/gpt-4o_cot/structured_data_results_filter1.json'
# file2 = '/data/liangzhuowen/projects/DocumentAI/results_prompt1/LegalBench/gpt-4o_cot/legalbench.json'
# file1 = '/data/liangzhuowen/projects/DocumentAI/results_prompt1/TATQA/train/gpt-4o_cot/structured_data_results_filter2_false.json'
# file2 = '/data/liangzhuowen/projects/DocumentAI/results_prompt1/Finqa/train/gpt-4o_cot/structured_data_results_filter2_false_wans.json'
# update_steps_from_cot(file1)  # 直接更新原文件
processed_results = process_json_file(file1, file2)


json_file = "/data/liangzhuowen/projects/DocumentAI/results_prompt1/Finqa/train/gpt-4o_cot/structured_data_results_filter2_update.json"
output_file = "/data/liangzhuowen/projects/DocumentAI/results_prompt1/Finqa/train/gpt-4o_cot/structured_data_results_filter2_update_true.json"
# filter_json(json_file, output_file, 3)




