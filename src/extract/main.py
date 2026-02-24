from src.structure_analysis.query2schema import get_schema, get_schema2
from src.extract.to_desc import to_desc
from src.extract.to_graph import to_graph, GraphMerger
from src.extract.to_table import to_table, TableMerger
from src.prompt import PROMPTS
import pandas as pd
import json

import time
import asyncio


def load_schema(json_file_path):
    """加载 schema.json 文件，返回 id 到 entity_type 的映射字典"""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            schema_list = json.load(f)
        
        # 将列表转换为字典格式 {id: entity_type}
        schema_dict = {}
        for item in schema_list:
            if isinstance(item, dict):
                for id_, entity_type in item.items():
                    schema_dict[id_] = entity_type
        return schema_dict
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading schema.json: {e}")
        return {}

def em_process(id, doc, input_text, question, data_structure, if_chunk, if_document):
    """
    Process the input text based on the data structure and the question.
    Args:
        id (str): The id of the data.
        doc (list): The document.
        input_text (str): The input text.
        question (str): The question.
        data_structure (str): The data structure.
        if_chunk (bool): Whether to process the input text in chunks.
        if_document (bool): Whether to process the input text as a document.
    Returns:
        list: A list of dictionaries containing the structured data, answer, cot process, latency, cot length, and schema.
    """
    while True:
        # schema = get_schema(question)
        # entity_type = [item.strip().strip("'") for item in schema.strip("()").split(',')]

        schema_dict = load_schema("../schema.json")
        entity_type = schema_dict.get(id, [])
        print(entity_type)

        if entity_type:
            break
        else:
            print("Empty entity_type detected, retrying...")

    if if_document:
        if 'graph' in data_structure:
            docs, titles = split_content_and_tile(input_text, doc)
            graphs = []

            merger = GraphMerger()
            for d, doc in enumerate(docs):
                print(f"data_id: {id}, do_construct_graph... in doc {d+1}/{len(docs)} in docs ..")
                title = titles[d] 
                content = doc['document']

                output = to_graph(content, entity_type)
                output['id'] = id
                output['doc'] = title
                output['data_structure'] = data_structure
                graphs.append(output)
                merger.add_graph(output["structured_data"])

            merged_graph = merger.get_edges()
            graphs.append({"merged_graph": merged_graph})
            return graphs

        elif 'table' in data_structure:
            docs, titles = split_content_and_tile(input_text, doc)
            tables = asyncio.run(process_tables_async(docs, titles, id, entity_type, data_structure))
            return tables

        else:
            return to_desc(input_text, question)

    elif if_chunk:
        if 'table' in data_structure:
            chunks = split(input_text, max_tokens=2048)
            print(len(chunks))
            tables = asyncio.run(process_chunks_async(chunks, id, entity_type, data_structure))
            return tables

        else:
            return to_desc(input_text, question)

    else:
        structure_fragments = input_text
        structured_data = to_structure(structure_fragments, entity_type, data_structure)
        return structured_data
    
def to_structure(structure_fragment, schema, data_structure):
    """
    Convert the input text into a structured data based on the data structure and the schema.
    Args:
        structure_fragment (str): The input text.
        schema (str): The schema.
        data_structure (str): The data structure.
    Returns:
        dict: A dictionary containing the structured data.
    """
    if 'graph' in data_structure:
        structured_data = to_graph(structure_fragment, schema)
    elif 'table' in data_structure:
        structured_data = to_table(structure_fragment, schema)
    else:
        structured_data = to_desc(structure_fragment, schema)
    return structured_data


def split_content_and_tile(docs_, doc):
    """
    Split the input text into content and title.
    Args:
        docs_ (str): The input text.
        doc (list): The document.
    Returns:
        list: A list of dictionaries containing the title and content.
    """
    docs = []
    titles = [] # document filenames
    
    raw_doc_list = docs_.strip("<标题起始符>").split("<标题起始符>")
    print(len(raw_doc_list))
    for idx, raw_doc in enumerate(raw_doc_list):
        if idx < len(doc):  # Ensure the title list does not exceed
            title = doc[idx]  # Use the title from the passed doc list
            content = raw_doc.split('<标题终止符>')[1].strip()  # Extract content
            
            docs.append({'title': title, 'document': content})

async def process_tables_async(docs, titles, id, entity_type, data_structure):
    """
    Process the input text into a table.
    Args:
        docs (list): The document.
        titles (list): The title.
        id (str): The id of the data.
        entity_type (str): The entity type.
        data_structure (str): The data structure.
    Returns:
        list: A list of dictionaries containing the table.
    """
    merger = TableMerger()
    tables = []

    async def process_one_doc(d_idx, doc_item, title):
        print(f"data_id: {id}, do_construct_table... in doc {d_idx+1}/{len(docs)} in docs ..")
        content = doc_item['document']
        output = await to_table(content, entity_type)
        output['id'] = id
        output['doc'] = title
        output['data_structure'] = data_structure
        return output

    tasks = [process_one_doc(d_idx, doc_item, titles[d_idx]) for d_idx, doc_item in enumerate(docs)]
    
    start_time = time.time()
    outputs = await asyncio.gather(*tasks)
    latency = time.time() - start_time

    for output in outputs:
        tables.append(output)
        merger.add_table(output['answer'], PROMPTS)

    try:
        merged_table1 = merger.merge(join_type="outer")
        if isinstance(merged_table1, pd.DataFrame):
            merged_table1 = merged_table1.to_markdown(index=False)
    except Exception as e:
        print(f"Merge failed: {e}")
        merged_table1 = ""

    try:
        merged_table2 = merger.union()
        if isinstance(merged_table2, pd.DataFrame):
            merged_table2 = merged_table2.to_markdown(index=False)
    except Exception as e:
        print(f"Union failed: {e}")
        merged_table2 = ""

    tables.append({"joined_table": merged_table1})
    tables.append({"unioned_table": merged_table2})
    tables.append({"ie_time": latency})

    return tables


async def process_chunks_async(chunks, id, entity_type, data_structure):
    """
    Process the input text into a table.
    Args:
        chunks (list): The document.
        id (str): The id of the data.
        entity_type (str): The entity type.
        data_structure (str): The data structure.
    Returns:
        list: A list of dictionaries containing the table.
    """
    merger = TableMerger()
    tables = []

    async def process_one_chunk(idx, chunk_text):
        output = await to_table(chunk_text, entity_type)
        output['id'] = id
        output['chunk_id'] = id
        output['data_structure'] = data_structure
        return output

    tasks = [process_one_chunk(idx, chunk) for idx, chunk in enumerate(chunks)]
    outputs = await asyncio.gather(*tasks)

    for output in outputs:
        tables.append(output)
        merger.add_table(output['answer'], PROMPTS)

    try:
        merged_table1 = merger.merge(join_type="outer")
        if isinstance(merged_table1, pd.DataFrame):
            merged_table1 = merged_table1.to_markdown(index=False)
    except Exception as e:
        print(f"Merge failed: {e}")
        merged_table1 = ""

    try:
        merged_table2 = merger.union()
        if isinstance(merged_table2, pd.DataFrame):
            merged_table2 = merged_table2.to_markdown(index=False)
    except Exception as e:
        print(f"Union failed: {e}")
        merged_table2 = ""

    tables.append({"joined_table": merged_table1})
    tables.append({"unioned_table": merged_table2})

    return tables






