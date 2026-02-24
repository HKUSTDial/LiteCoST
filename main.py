"""
LiteSEA Main Program
Purpose: Generate Chain-of-Structured-Thought Data from unstructured data
Main functions:
- Document processing and information extraction
- Structure analysis and reasoning
"""

import json
import os
import re
import random
import time

import llm

from src.structure_analysis.structure_decision import select  # Args: text, Returns: structure type
from src.reasoner import reasoning, parse_answer_r1  # Args: query,context, Returns: reasoning result
from src.extract.main import em_process  # Args: text, Returns: extraction result
from src.extract.to_desc import to_desc  # Args: text, Returns: text description
from src.extract.graph import Graph  # Args: nodes,edges, Returns: graph object
from src.utils import *  # Utility functions collection

from src.prompt import PROMPTS  # Prompt templates
import llm.global_config as config  # Global configurations


def generate_prompt_Loong(item):
    """
    Generate a prompt based on the input data from Loong.

    Args:
        data (dict): A dictionary containing the necessary fields for prompt generation.

    Returns:
        tuple: A tuple containing the question, documents, and evidence.
    """
    replace_dict = {"{question}": item['question'], "{instruction}": item['instruction']}
    prompt_template = item['prompt_template']
    doc = item['doc']
    # evidence = item['evidence']
    evidence = item['docs']
    
    for k, v in replace_dict.items():
        prompt_template = prompt_template.replace(k, v)

    if isinstance(evidence, list):  # 如果 evidence 是列表，则用换行符连接
        evidence = "\n".join(evidence)
    return prompt_template, evidence, doc


def generate_prompt_Financebench(data):
    """
    Generate a prompt based on the input data from Financebench.

    Args:
        data (dict): A dictionary containing the necessary fields for prompt generation.

    Returns:
        tuple: A tuple containing the question and documents.
    """
    # Extract relevant fields from the input data
    evidence = data.get("evidence", [])
    docs = ""
    if evidence:
        docs = "\n".join(e["evidence_text"] for e in evidence)

    question = data.get("question", "")
    question_type = data.get("question_type","")
    question_reasoning = data.get("question_reasoning","")

    return question, docs

def generate_prompt_Finqa(data):
    """
    Generate a prompt based on the input data from Finqa.

    Args:
        data (dict): A dictionary containing the necessary fields for prompt generation.

    Returns:
        tuple: A tuple containing the question and documents.
    """
    # Extract relevant fields from the input data
    docs = data.get("context", "")
    evidence = data.get("gold_evidence", "")
    question = data.get("question", "")

    return question, docs

def generate_prompt_TAT(data):
    """
    Generate a prompt based on the input data from TATQA.

    Args:
        data (dict): A dictionary containing the necessary fields for prompt generation.

    Returns:
        tuple: A tuple containing the question and documents.
    """
    # Extract relevant fields from the input data
    docs = data.get("context", "")
    question = data.get("question", "")

    return question, docs

# 返回id和save id
def get_record_id(record, dataset):
    """
    Get the record ID and save ID based on the dataset type.

    Args:
        record (dict): A dictionary containing the record data.
        dataset (str): The dataset type.

    Returns:
        tuple: A tuple containing the record ID and save ID.
    """
    dataset = dataset.lower()

    if dataset == 'finqa':
        return record['id'], record["id"].replace("/", "_")
    elif dataset == 'financebench':
        return record["financebench_id"], record["financebench_id"]
    elif dataset in ['loong', 'loongfin',' tatqa']:
        return record["id"], record["id"]
    else:
        raise ValueError(f"Unsupported dataset type: {dataset}")


def run_process(dataset, model, if_structured, if_chunk, if_document, qa_file, ds_file, ds_folder, structured_data_file, structured_data_folder, data_vis_folder, result_data_folder, result_file):
    """
    Run the process for the given dataset and model to generate Chain-of-Structured-Thought Data from unstructured data.

    Args:
        dataset (str): The dataset type.
        model (str): The model name.
        if_structured (bool): Whether to use structured data.
        if_chunk (bool): Whether to use chunk data.
        if_document (bool): Whether to use document data.
        qa_file (str): The path to the question and answer file.
        ds_file (str): The path to the data selection file.
        ds_folder (str): The path to the data selection folder.
        structured_data_file (str): The path to the structured data file.
        structured_data_folder (str): The path to the structured data folder.
        data_vis_folder (str): The path to the data visualization folder.
        result_data_folder (str): The path to the result data folder.
        result_file (str): The path to the result file.
    """
    
    prompt_generator_map = {
        'finqa': generate_prompt_Finqa,
        'loong': generate_prompt_Loong,
        'loongfin': generate_prompt_Loong,
        'tatqa': generate_prompt_TAT ,
        'financebench': generate_prompt_Financebench
    }
    generate_prompt = prompt_generator_map.get(dataset.lower())
    
    create_folder(ds_folder)
    create_folder(result_data_folder)
    create_folder(structured_data_folder)
    create_folder(data_vis_folder)

    # Read the data
    qa_datas = []
    with open(qa_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                record = json.loads(line.strip())
                qa_datas.append(record)
    print(111)

    if if_structured:
        # Data Structure Selection
        ds_datas = {}
        if os.path.exists(ds_file):
            ds_records = read_json(ds_file)
            for record in ds_records:
                ds_datas[str(record["id"])] = record["data_structure"]
        else:
            ds_results = []
            for record in qa_datas:
                id, saved_id = get_record_id(record, dataset)
                batch_file = f"{ds_folder}/{saved_id}.json"

                if os.path.exists(batch_file):
                    continue
                
                question,input_text = generate_prompt(record)
                # print(input_text)
                ds, explain = select(question, need_explain=True)
                print(ds, explain)
                result = {'id': id,
                        'question': question,
                        'answer': record['answer'],
                        'data_structure': ds,
                        'explanation': explain}
                ds_results.append(result)
                ds_datas[str(id)] = ds

                save_to_json(ds_results, batch_file)
                ds_results.clear()
            merge_json_files(ds_folder, ds_file)
            # save_to_json(ds_results, ds_file)
        print(222)

        # Structured Data Extraction
        structured_datas = []

        if os.path.exists(structured_data_file):
            structured_datas = read_json(structured_data_file)
        else:
            for i, record in enumerate(qa_datas):        
                id, saved_id = get_record_id(record, dataset)
                batch_file = f"{structured_data_folder}/{saved_id}.json"

                if os.path.exists(batch_file):
                    continue
                # if i==2: break
                # if id != "2c0d2c6b-6646-4598-b126-e7c6d25459a4": continue
                if dataset.lower() == 'loong' or dataset.lower() == 'loongfin':
                    question,input_text,doc = generate_prompt(record)
                else:
                    question,input_text = generate_prompt(record)
                    
                ds = ds_datas.get(id, "description")
                context_length = token_length(input_text)
                
                if 'Table' in ds: ds = 'Table' 
                elif 'Graph' in ds: ds = 'Graph'
                elif 'Tree' in ds: ds = 'Tree'
                else: ds = 'Text Description'
                
                print(i, id, ds, context_length)

                if ds == "Text Description": 
                    result = {'id': id,
                            'question': question,
                            'data_structure': ds,
                            'structured_data': input_text,
                            'cot': input_text,
                            }
                    structured_datas.append(result)
                    save_to_json(structured_datas, batch_file)
                    structured_datas.clear()
                else:
                    if if_document or if_chunk:
                        structured_data = em_process(id=id, doc=doc, input_text=input_text, question=question, data_structure=ds.lower(), if_chunk=if_chunk, if_document=if_document)
                        save_to_json(structured_data, batch_file)
                    else:
                        structured_data = em_process(id=id, doc=doc, input_text=input_text, question=question, data_structure=ds.lower(), if_chunk=if_chunk, if_document=if_document)

                        if structured_data:
                            result = {'id': id,
                                    'question': question,
                                    'data_structure': ds,
                                    'schema': structured_data['schema'],
                                    'structured_data': structured_data['structured_data'],
                                    'answer': structured_data['answer'],
                                    # 'steps': structured_data['steps'],
                                    'cot': structured_data['cot'],
                                    'cot_length': structured_data['cot_length'],
                                    'latency': structured_data['latency'],
                                    }
                        structured_datas.append(result)
                        save_to_json(structured_datas, batch_file)
                        structured_datas.clear()
                        time.sleep(5)
            

        # Visualize
        structured_datas = read_json(structured_data_file)
        print(len(structured_datas))

        for record in structured_datas:
            ds = record.get('data_structure', "")
            structured_data = record.get('structured_data', "") 
            file_name = f"{record['id']}"  # 使用 record 的 id 作为文件名 
            file_name = file_name.replace("/", "_") if dataset.lower() == 'finqa' else file_name

            if ds == "Table":
                with open(os.path.join(data_vis_folder, f"{file_name}.md"), 'w', encoding='utf-8') as file:
                    file.write(structured_data)
            elif ds == 'Graph':
                G = Graph()
                G.create_graph_from_triplets(structured_data)
                visualization = G.generateGraph()
                with open(os.path.join(data_vis_folder, f"{file_name}.html"), "w", encoding="utf-8") as file:
                    file.write(visualization)
            else:
                with open(os.path.join(data_vis_folder, f"{file_name}.txt"), 'w', encoding='utf-8') as file:
                    file.write(structured_data)
        # exit()

        # Reasoning
        final_results = []
        check_result = False
        if os.path.exists(result_file):
            final_results = read_json(result_file)
        else:
            for record in qa_datas:
                id, saved_id = get_record_id(record, dataset)
                record_file = f"{result_data_folder}/{saved_id}.json"
                if os.path.exists(record_file):
                    continue

                std_ans = record['answer']
                question = record["question"]
                
                # if id != "2c0d2c6b-6646-4598-b126-e7c6d25459a4": continue
                print(id)
                if if_chunk or if_document:
                    structured_datas = read_json_by_id(id, structured_data_folder)
                    ds = structured_datas[0]["data_structure"]
                    if ds == "Text Description": continue
                    # structured_data = '\n'.join([item['structured_data'] for item in structured_datas if 'structured_data' in item])
                   
                    # If structured_data is not available, use cot instead
                    structured_data = '\n'.join([item.get('structured_data') if item.get('structured_data') else item.get('cot', '') for item in structured_datas])


                else:
                    structured_data = next((sd for sd in structured_datas if sd["id"] == id), None)
                    if not structured_data: continue

                    # structured_data = structured_data['structured_data']
                    structured_data = structured_data['cot'] #cot-structure

                print(id, question)
                print(structured_data)

                start_time = time.time()
                cot_ans, extracted_ans = reasoning(question, structured_data, model='yizhan')
                latency = time.time() - start_time

                final_ans = extract_answer_content(parse_answer_r1(cot_ans))
                # check_result = check_answer(question, final_ans, std_ans)
                print(f"answer: {final_ans}")
                # print(check_result)
                print("---------------")

                final_results.append({
                    "id": record["id"],
                    "question": question,
                    'cot_answer': cot_ans,
                    'answer': final_ans,
                    'extracted_answer': extracted_ans,
                    'std_answer': std_ans,
                    'cot_length': token_length(cot_ans),
                    'latency': latency
                    # 'check_answer': check_result
                })
                save_to_json(final_results, record_file)
                final_results.clear()
            # merge_json_files(result_data_folder, result_file)
    else:
        print(444)
        final_results = []
        check_result = False
        for i, record in enumerate(qa_datas):
            # if i==2: break 
            id, saved_id = get_record_id(record, dataset)
            record_file = f"{result_data_folder}/{saved_id}.json"
            # if os.path.exists(record_file):
            #     continue

            # if record["id"]!="id_01484": continue
            if dataset.lower() == 'loong':
                question,input_text,doc = generate_prompt(record)
            else:
                question,input_text = generate_prompt(record)
            std_ans = record['answer']

            print(id, question, token_length(input_text))

            # if token_length(input_text)>65536:
            #     if os.path.exists(record_file):
            #         os.remove(record_file)
            #         print(f"File {record_file} deleted due to excessive input length.")
            start_time = time.time()
            cot_ans, extracted_ans = reasoning(question, input_text, model='yizhan')
            latency = time.time() - start_time

            final_ans = extract_answer_content(parse_answer_r1(cot_ans))
            # check_result = check_answer(question, final_ans, std_ans)
            print(i, final_ans)
            # print(check_result)
            print("---------------")

            final_results.append({
                "id": record["id"],
                "question": question,
                'cot_answer': cot_ans,
                'answer': final_ans,
                'extracted_answer': extracted_ans,
                'std_answer': std_ans,
                'cot_length': token_length(cot_ans),
                'latency': latency
                # 'check_answer': check_result
            })
            
            save_to_json(final_results, record_file)
            final_results.clear()
        merge_json_files(result_data_folder, result_file)
if __name__ == '__main__':
    # python main.py --model gpt-4 --dataset Loong --structured --document
    import argparse
    
    parser = argparse.ArgumentParser(description='Run document processing pipeline')
    
    # Add command line arguments
    parser.add_argument('--model', type=str, default='gpt-4o', help='Choose model (gpt, llama, deepseek)')
    parser.add_argument('--dataset', type=str, default='Loong', help='Dataset name (finqa, financebench, tatqa, loong)')
    parser.add_argument('--structured', action='store_true', default=True, help='Whether to use structured processing')
    parser.add_argument('--chunk', action='store_true', default=False, help='Whether to process in chunks')
    parser.add_argument('--document', action='store_true', default=True, help='Whether to process documents')
    
    args = parser.parse_args()
    
    # Set model
    config.set_model(args.model)
    print(config.get_model())
    
    # Build file paths
    qa_file = f'./dataset/{args.dataset}/loong_process.jsonl'
    ds_file = f'./results/{args.dataset}/data_selection_results.json'
    ds_folder = f'./results/{args.dataset}/structures'
    structured_data_file = f'./results/{args.dataset}/structured_data_results.json'
    structured_data_folder = f'./results/{args.dataset}/data_structure_results'
    data_vis_folder = f'./results/{args.dataset}/data_vis'
    result_data_folder = f'./results/{args.dataset}/{args.model}/ours_results_{args.model}' if args.structured else f'./results/{args.dataset}/{args.model}/llm_results'
    result_file = f"./results/{args.dataset}/{args.model}/ours_{args.model}.json" if args.structured else f"./results/{args.dataset}/{args.model}/llm_{args.model}.json"
    
    run_process(args.dataset, args.model, args.structured, args.chunk, args.document, 
                qa_file, ds_file, ds_folder, structured_data_file, structured_data_folder,
                data_vis_folder, result_data_folder, result_file)
