"""
Data Refinement: this script is used to refine the structured data.
"""
import json
import pandas as pd
import os
from utils import *
from extract.to_table import regenerate_table

# Table Refinement
def process_and_update_table(qa_file, json_file, updated_folder, structured_data_file, MAX_RECHECK_ATTEMPTS=2):
    """
    Process and update table

    Args:
        qa_file (str): Path to input JSON file
        json_file (str): Path to input JSON file
        updated_folder (str): Path to output folder
        structured_data_file (str): Path to input JSON file
        MAX_RECHECK_ATTEMPTS (int): Maximum number of attempts to regenerate the table
    """
    create_folder(updated_folder)
    
    # Read JSON file
    data = read_json(json_file)
    
    qa_datas = []
    with open(qa_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                record = json.loads(line.strip())
                qa_datas.append(record)

    updated_ress = []
    # Extract information from JSON data
    for item in data:
        id = item['id']
        save_id = id.replace("/", "_")

        question = item["question"]
        current_table = item["answer"]
        data_structure = item["data_structure"]
        content = next((sd for sd in qa_datas if sd["id"] == id), None)['context']
        batch_file = f"{updated_folder}/{save_id}.json"

        if os.path.exists(batch_file):
            continue

        print(id, question)
        print(current_table)
        # print(content)
        # Call regenerate_table function to update
        updated_res, need_recheck = regenerate_table(MAX_RECHECK_ATTEMPTS, question, current_table, content)
        
        
        if need_recheck:
            # Output updated results
            updated_res['id'] = id
            updated_res['question'] = question
            updated_res['data_structure'] = data_structure
            print(f"Updated Response:\n{updated_res}")
            updated_ress.append(updated_res)

        else:
            item['need_recheck'] = need_recheck
            updated_ress.append(item)

        # break

        save_to_json(updated_ress, batch_file)
        updated_ress.clear()
    # merge_json_files(updated_folder, structured_data_file)