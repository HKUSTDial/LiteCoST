"""
Convert data to GPT conversation / Aplaca instruction format.
"""
from utils import *

import json, os
from sklearn.model_selection import train_test_split


instruction_mapper = {
    'Table': "You are an expert in table construction. Given a user query, input data and a target table schema, your task is to extract and map cells from the input to each schema column row by row to build complete, relevant tables. Ensure that the extracted table is relevant to the user's question. If a schema column has no corresponding data, return an empty list for that column. Please provide your output as a Markdown-formatted table.",
    'Graph': "You are an expert in graph construction. Your task is to analyze input data, user queries, and a provided schema to extract meaningful relationships between entities. You should structure these relationships as logical triplets [head, relation, tail], ensuring that both the head and tail belong to the schema and the extracted relationships follow a connected structure and are relevant to the user's question. If no relevant relationships exist, return an empty list. Please provide your output as a JSON-formatted string.",

    'Table' + 'cot': "You are an expert in table construction. Please extract entities that match the schema definition from the input, and finally generate the structured table.",
    'Graph' + 'cot': "You are an expert in graph construction. Please extract relationship triples that match the schema definition from the input, and finally generate the structured graph.",

    'Table' + 'cot' + "attribute": "You are an expert in table construction. Please extract entities that match the schema definition and its corresponding attribute mapping from the input, and finally generate the structured table.",
    'Graph' + 'cot' + "attribute": "You are an expert in graph construction. Please extract relationship triples that match the schema definition and its corresponding attribute mapping from the input, and finally generate the structured graph.",
}

SYSTEM_PROMPT = """
Please reason step by step, and respond in the following format: 
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""



def construct_conversation_train(json_file_1, json_file_2, output_file1, output_file2, if_cot=False):
    """
    Read JSON file and convert data to GPT conversation format.
    Args:
        json_file_1: str, path to the first JSON file
        json_file_2: str, path to the second JSON file
        output_file1: str, path to the output file for SFT
        output_file2: str, path to the output file for GRPO
        if_cot: bool, whether to use COT
    """
    # Read JSON file
    with open(json_file_1, 'r', encoding='utf-8') as file:
        data1 = [json.loads(line.strip()) for line in file if line.strip()] if json_file_1.endswith('.jsonl') else json.load(file)
    
    with open(json_file_2, 'r', encoding='utf-8') as file:
        data2 = json.load(file)
    
    # Construct ID mapping
    schema_map = {item["id"]: item.get("schema", []) for item in data2}
    data_structure_map = {item["id"]: item.get("data_structure", "") for item in data2}
    structure_data_map = {item["id"]: item.get("structured_data", "") for item in data2}
    steps_map = {item["id"]: item.get("steps", []) for item in data2}
    cot_map = {item["id"]: item.get("cot", "") for item in data2}
    answer_map = {item["id"]: item.get("answer", "") for item in data2}

    sft_result = []
    grpo_result = []
    count = 0  # Count invalid data

    for item in data1:        
        # financebench
        # id_ = item["financebench_id"]
        # evidence = item.get("evidence", [])
        # docs = ""
        # if evidence:
        #     docs = "\n".join(e["evidence_text"] for e in evidence)
        # context = docs
        # query = item["question"]

        # Finqa / TATQA
        id_ = item["id"]
        context = item["context"]
        query = item["question"]
        
        schema = schema_map.get(id_, [])
        data_structure = data_structure_map.get(id_, "")
        structured_data = structure_data_map.get(id_, "")
        steps = steps_map.get(id_, [])
        cot = cot_map.get(id_, "")
        answer = answer_map.get(id_, "")
        
        # if not data_structure or data_structure == "Text Description" or not schema or not steps:
        #     continue
        if not data_structure or data_structure == "Text Description" or not schema:
            continue
        
        if not isinstance(steps, list) and data_structure == "Table":
            count += 1
            continue
        
    
        # Generate conversation format
        sft_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps({"instruction": instruction_mapper[data_structure + 'cot'] if if_cot else instruction_mapper[data_structure], "schema": schema, "input": context}, ensure_ascii=False)},
            {"role": "assistant", "content": cot if if_cot else json.dumps(structured_data, ensure_ascii=False)}
        ]

        grpo_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps({"instruction": instruction_mapper[data_structure + 'cot'] if if_cot else instruction_mapper[data_structure], "schema": schema, "input": context}, ensure_ascii=False)}
        ]
        
        sft_result.append({"messages": sft_messages})
        grpo_result.append({
            "prompt": grpo_messages,
            "answer": answer
        })
    
    with open(output_file1, 'w', encoding='utf-8') as file:
        json.dump(sft_result, file, indent=4, ensure_ascii=False)
    with open(output_file2, 'w', encoding='utf-8') as file:
        json.dump(grpo_result, file, indent=4, ensure_ascii=False)
    
    print(f"Filtered invalid steps count: {count}")

def construct_conversation_test(json_file_1, json_file_2, output_file, if_cot=False, if_steps=False):
    """
    Read JSON file and convert data to GPT conversation format.
    Args:
        json_file_1: str, path to the first JSON file
        json_file_2: str, path to the second JSON file
        output_file: str, path to the output file
        if_cot: bool, whether to use COT
    """
    # Read JSON file
    if json_file_1.endswith('.jsonl'):
        data1 = []
        with open(json_file_1, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    record = json.loads(line.strip())
                    data1.append(record)
    else:
        with open(json_file_1, 'r', encoding='utf-8') as file:
            data1 = json.load(file)  # Format like [{"id": "xxx", "query": "", "context": "some text"}]

    with open(json_file_2, 'r', encoding='utf-8') as file:
        data2 = json.load(file)  # Format like [{"id": "xxx", "schema": ["person", "organization"]}, "data_structure": "Table", "structured_data": {...}}]


    # Construct ID mapping, convenient to find
    schema_map = {item["id"]: item.get("schema", []) for item in data2}
    data_structure_map = {item["id"]: item.get("data_structure", "") for item in data2}
    structure_data_map = {item["id"]: item.get("structured_data", "") for item in data2}
    steps_map = {item["id"]: item.get("steps", []) for item in data2}
    cot_map = {item["id"]: item.get("full_cot", "") for item in data2}
    answer_map = {item["id"]: item.get("answer", "") for item in data2}

    sft_result = []    

    for item in data1:
        id_ = item["id"]
        
        # financebench
        # evidence = item.get("evidence", [])
        # docs = ""
        # if evidence:
        #     docs = "\n".join(e["evidence_text"] for e in evidence)
        # context = docs

        # finqa
        context = item["context"]
        query = item["question"]
        
        # Get schema and data_structure
        schema = schema_map.get(id_, [])
        data_structure = data_structure_map.get(id_, "")
        structured_data = structure_data_map.get(id_, "")
        steps = steps_map.get(id_, [])
        cot = cot_map.get(id_, "")
        answer = answer_map.get(id_, "")

        if data_structure == "Text Description": continue

        # Generate conversation format
        sft_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps({"instruction": instruction_mapper[data_structure + 'cot'] if if_cot else instruction_mapper[data_structure], "schema": schema, "input": context}, ensure_ascii=False)},
            {"role": "assistant", "content": cot if if_cot else json.dumps(structured_data, ensure_ascii=False)}
        ]
        
        sft_result.append({"messages": sft_messages})
        
    # Write results to JSON file
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(sft_result, file, indent=4, ensure_ascii=False)
    # with open(output_file, 'w', encoding='utf-8') as writer:
    #     for data in result:
    #         writer.write(json.dumps(data, ensure_ascii=False)+"\n")

def construct_instruction_train(json_file_1, json_file_2, output_file, if_cot=False, if_steps=False):
    """
    Read JSON file and convert data to instruction format.
    Args:
        json_file_1: str, path to the first JSON file
        json_file_2: str, path to the second JSON file
        output_file: str, path to the output file
        if_cot: bool, whether to use COT
        if_steps: bool, whether to use steps
    """
    # Read JSON file
    if json_file_1.endswith('.jsonl'):
        data1 = []
        with open(json_file_1, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    record = json.loads(line.strip())
                    data1.append(record)
    else:
        with open(json_file_1, 'r', encoding='utf-8') as file:
            data1 = json.load(file)  # Format like [{"id": "xxx", "query": "", "context": "some text"}]

    with open(json_file_2, 'r', encoding='utf-8') as file:
        data2 = json.load(file)  # Format like [{"id": "xxx", "schema": ["person", "organization"]}, "data_structure": "Table", "structured_data": {...}}]

    # Construct ID mapping, convenient to find
    schema_map = {item["id"]: item.get("schema", None) for item in data2}
    data_structure_map = {item["id"]: item["data_structure"] for item in data2}
    structure_data_map = {item["id"]: item["structured_data"] for item in data2}
    steps_map = {item["id"]: item.get("steps", None) for item in data2}
    cot_map = {item["id"]: item["cot"] for item in data2}
    # structure_map = {item["id"]: (item["data_structure"], item["structured_data"]) for item in data2}
    
    result = []
    count=0 # Count data without correct steps, and filter

    for item in data1:
        # id_ = item["financebench_id"]
        
        # # financebench
        # evidence = item.get("evidence", [])
        # docs = ""
        # if evidence:
        #     docs = "\n".join(e["evidence_text"] for e in evidence)
        # context = docs
        # query = item["question"]

        # finqa
        id_ = item["id"]
        context = item["context"]
        query = item["question"]

        # Get schema and data_structure
        schema = schema_map.get(id_, [])
        data_structure = data_structure_map.get(id_, "")
        structure_data = structure_data_map.get(id_, "")
        steps = steps_map.get(id_, [])
        cot = cot_map.get(id_, "")
        # data_structure, structured_data = structure_map.get(id_, (None, None))
        if not data_structure: continue
        # if data_structure == "Text Description" or not schema or not steps: continue
        if data_structure == "Text Description" or not schema: continue

        # if not isinstance(steps, list) and data_structure == "Table" : 
        #     count+=1
        #     continue

        # Generate conversation format
        if if_cot:
            if if_steps:
                pass
            else:
                instruction_text = instruction_mapper[data_structure + 'cot']

                sinstruct = {
                    "instruction": instruction_text,
                    # "query": query,
                    "schema": schema,
                }

                instruction = {
                    "instruction": json.dumps(sinstruct, ensure_ascii=False), 
                    "input": context,
                    "output": cot,
                    'system': SYSTEM_PROMPT 
                    # "system": "Please reason step by step, and put your final answer followed by **Final Output Formatting:**."
                }
        else:
            # Generate different instructions
            instruction_text = instruction_mapper[data_structure]
            sinstruct = {
                "instruction": instruction_text,
                # "query": query,
                "schema": schema,
            }

            instruction = {
                "instruction": json.dumps(sinstruct, ensure_ascii=False), 
                "input": context,
                "output": str(structured_data)
            }

        # print(instruction)
        result.append(instruction)

    # Write results to JSON file
    # with open(output_file, 'w', encoding='utf-8') as file:
    #     json.dump(result, file, indent=4, ensure_ascii=False)
    with open(output_file, 'w', encoding='utf-8') as writer:
        for data in result:
            writer.write(json.dumps(data, ensure_ascii=False)+"\n")
    
    print(count)
    return result

# jsonl -> json
def construct_instruction_test(json_file_1, json_file_2, output_file):
    """
    Read JSON file and convert data to instruction format.
    Args:
        json_file_1: str, path to the first JSON file
        json_file_2: str, path to the second JSON file
        output_file: str, path to the output file
    """
    if json_file_1.endswith('.jsonl'):
        data1 = []
        with open(json_file_1, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    record = json.loads(line.strip())
                    data1.append(record)
    else:
        with open(json_file_1, 'r', encoding='utf-8') as file:
            data1 = json.load(file)  # Format like [{"id": "xxx", "query": "", "context": "some text"}]

    with open(json_file_2, 'r', encoding='utf-8') as file:
        data2 = json.load(file)  # Format like [{"id": "xxx", "schema": ["person", "organization"]}, "data_structure": "Table", "structured_data": {...}}]


    # Construct ID mapping, convenient to find
    data_structure_map = {item["id"]: item["data_structure"] for item in data2}
    schema_map = {item["id"]: item.get("schema", None) for item in data2}
    cot_map = {item["id"]: item["cot"] for item in data2}
    
    result = []

    for item in data1:
        id_ = item["id"]
        
        # financebench
        # evidence = item.get("evidence", [])
        # docs = ""
        # if evidence:
        #     docs = "\n".join(e["evidence_text"] for e in evidence)
        # context = docs

        # finqa
        context = item["context"]
        query = item["question"]
        
        # Get schema and data_structure
        schema = schema_map.get(id_, [])
        data_structure = data_structure_map.get(id_, "")
        cot = cot_map.get(id_, "")
        if data_structure == "Text Description": continue

        # Generate conversation format
        instruction_text = instruction_mapper[data_structure + 'cot']

        sinstruct = {
            "instruction": instruction_text,
            "schema": schema,
        }

        instruction = {
            "instruction": json.dumps(sinstruct, ensure_ascii=False), 
            "input": context,
            "output": cot,
            "system": SYSTEM_PROMPT
            # "system": "Please reason step by step, and put your final answer followed by **Final Output Formatting:**."
        }

        # print(instruction)
        result.append(instruction)

    # Write results to JSON file
    # with open(output_file, 'w', encoding='utf-8') as file:
    #     json.dump(result, file, indent=4, ensure_ascii=False)
    with open(output_file, 'w', encoding='utf-8') as writer:
        for data in result:
            writer.write(json.dumps(data, ensure_ascii=False)+"\n")

    return result
        
if __name__ == "__main__":
    # Construct instruction dataset
    json_file_1 = "./dataset/Finqa/finqa_train.jsonl"  # Replace with your file path
    json_file_2 = "./results/Finqa/train/deepseek-r1/structured_data_results_filter2_update_true.json"
    output_file1 = "./results/Finqa/result_r1/finqa_train_gpt.json"
    output_file2 = "./results/Finqa/result_r1/finqa_train_grpo.json"
    output_file = "./results/Finqa/result_r1/finqa_train_instruction.json"
    construct_instruction_train(json_file_1, json_file_2, output_file, if_cot=True, if_steps=False)
    construct_conversation_train(json_file_1, json_file_2, output_file1, output_file2, if_cot=True)

    # construct_instruction_test(json_file_1, json_file_2, output_file)
    # construct_conversation_test(json_file_1, json_file_2, output_file)

