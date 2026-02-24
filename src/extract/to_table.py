"""
This module provides functionality for converting data into a table.
"""

import llm as llm
import time
import ast
import os, logging, re
import math, jieba, pickle
import pandas as pd
import numpy as np
from functools import reduce
import asyncio

from src.rm_prompt import RM_PROMPTS
from src.prompt import PROMPTS

from src.extract.table import Table
from src.structure_analysis.query2schema import get_schema
from src.seek.main import find_rel, split, recursvie_split
from src.reasoner import reasoning
from src.utils import *


ENCODER = None
jieba.setLogLevel(log_level=logging.INFO)



SYSTEM_PROMPT = """
Please reason step by step, and respond in the following format: 
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

def parse_table(response, tuple_delimiter, record_delimiter, completion_delimiter):
    """
    Parse the response from the LLM into a table.
    Args:
        response (str): The response from the LLM.
        tuple_delimiter (str): The delimiter for the tuple.
        record_delimiter (str): The delimiter for the record.
        completion_delimiter (str): The delimiter for the completion.
    Returns:
        tuple: A tuple containing the title, header, rows, and description of the table.
    """
    def strip_trailing_parenthesis(part):
        return part.rstrip(')")')

    # print(response)
    # exit()

    # 保存中间结果
    # steps = extract_intermediate_results(response)
    # print(steps)
    
    # if completion_delimiter in steps[0]:
    #     # print(1111111111111111111111)
    #     entries = steps[0].split(completion_delimiter)[0].split(record_delimiter)
    # elif completion_delimiter in steps[-1]:
    #     # print(2222222222222222222222)
    #     entries = steps[-1].split(completion_delimiter)[0].split(record_delimiter)
    # else:
    #     # print(3333333333333333333333)
    #     entries = response.split(completion_delimiter)[0].split(record_delimiter)
    
    entries = response.split(completion_delimiter)[0].split(record_delimiter)

    title = ""
    description = ""
    header = []
    rows = []
    # print("=====")
    
    for entry in entries:
        parts = entry.replace('\n', '').split(tuple_delimiter)

        if len(parts) < 2:
            print("Skipping due to insufficient parts")
            continue
        if "table" in parts[0]:
            if len(parts) < 3:
                print(f"Skipping table entry with insufficient parts: {entry}")
                continue
            title = parts[1]
            description = parts[2]
        elif "header" in parts[0]:
            header = [strip_trailing_parenthesis(part.strip('"'))  for part in parts[1:]]
        elif "row" in parts[0]:
            rows.append([strip_trailing_parenthesis(part.strip('"'))  for part in parts[1:]])
    return title, header, rows, description

async def process_table_generation_async(llm_answer_text, MAX_RETRIES=3):
    """
    Input the text returned by LLM, parse it into a table.
    Args:
        llm_answer_text (str): The text returned by LLM.
        MAX_RETRIES (int): The maximum number of retries.
    Returns:
        tuple: A tuple containing the structured data, answer, cot process, and success.
    """
    refined_data = ""
    process_tables = ""
    success = False

    for retry_count in range(MAX_RETRIES):
        try:
            print(f"\nAttempt {retry_count + 1}/{MAX_RETRIES}")

            last_tables = llm_answer_text

            print(last_tables)
            if "<think>" in last_tables:
                process_tables = re.sub(r"<think>.*?</think>", "", last_tables, flags=re.DOTALL).strip()
            elif "<reasoning>" in last_tables:
                process_tables = re.sub(r"<reasoning>.*?</reasoning>", "", last_tables, flags=re.DOTALL).strip()
            else:
                process_tables = last_tables

            process_tables = extract_answer_content(process_tables)

            title, header, rows, description = parse_table(
                process_tables,
                PROMPTS["DEFAULT_TUPLE_DELIMITER"],
                PROMPTS["DEFAULT_RECORD_DELIMITER"],
                PROMPTS["DEFAULT_COMPLETION_DELIMITER"]
            )
            print("----------")
            print(title)
            print(header)
            print(rows)
            print("-----")
            if not rows:  # ✅ 如果 rows 为空
                print("Rows are empty, fallback to raw process_tables")
                process_tables = ""
                refined_data = ""
            else:
                df = pd.DataFrame(rows, columns=header)
                refined_data = df.to_markdown(index=False)  # no index column
                refined_data = f"# {title}\n\n{refined_data}"

            success = True
            return refined_data, process_tables, last_tables, success

        except Exception as e:
            print(f"Parsing failed at attempt {retry_count + 1}, error: {e}")
            await asyncio.sleep(2)  # 短一点时间

    print("Fallback to raw LLM output after 3 retries")
    return refined_data, process_tables, llm_answer_text, success

async def to_table(data_list: list, entity_type: str, model='llama-3.1-8b-instruct') -> list:
    """
    Receive multiple data, batch construct prompts, batch send requests, and batch parse.
    Args:
        data_list (list): The list of data to process.
        entity_type (str): The entity type.
        model (str): The model to use.
    Returns:
        list: A list of dictionaries containing the structured data, answer, cot process, latency, cot length, and schema.
    """
    print(f"Received {len(data_list)} chunks to process.")
    prompts = []
    for data in data_list:
        examples = PROMPTS['TABLE_CONSTRUCTION_EXAMPLES']
        raw_prompt = RM_PROMPTS['TABLE_CONSTRUCTION_R1']

        # CoT prompt
        table_construction_prompt = raw_prompt.format(
            # examples=examples,
            schema=entity_type,
            # attribute=attribute,
            tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
            record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
            completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
            # query=question,
            content=data,
        )

        # Testing Prompt
        table_construction_prompt2 =f'''
            {{
            "instruction": "You are an expert in table construction. Please extract entities that match the schema definition from the input, and finally generate the structured table.",
            "schema": {entity_type},
            "input": "{data}"
            }}
        '''

        # raw Prompt
        table_construction_prompt3 = f"""
        You are an expert in table construction.
        Please extract entities that match the schema definition from the input text and generate a structured table.

        Schema:
        {entity_type}

        Input Text:
        {data}

        Please output the final table in a structured format.
        """
        prompts.append(table_construction_prompt3)

    # === 批量发送所有 prompt            
    answers, times = await llm.async_get_answer(prompts, model=model, system_prompt=None)

    results = []
    for answer, latency in zip(answers, times):
        try:
            refined_data, process_tables, last_tables, success = await process_table_generation_async(answer)

            if not success:
                print(f"Warning: parse_table failed at chunk, fallback to raw table output.")
                refined_data = last_tables

            results.append({
                "schema": entity_type,
                "structured_data": refined_data,
                "cot": last_tables,
                "cot_length": token_length(last_tables),
                "answer": process_tables,
                "latency": latency,
            })

        except Exception as e:
            print(f"Error processing chunk: {e}")
            continue

    return results

def regenerate_table(MAX_RECHECK_ATTEMPTS, question, current_table, data):
    """
    Regenerate the table based on the current table and the original data.
    Args:
        MAX_RECHECK_ATTEMPTS (int): The maximum number of retries.
        question (str): The question.
        current_table (str): The current table.
        data (str): The original data.
    Returns:
        tuple: A tuple containing the updated table, whether it needs to be rechecked, and the recheck count.
    """
    if_loop_prompt = PROMPTS["TABLE_IF_LOOP_PROMPT"].format(
            current_table=current_table,
            query=question,
            content=data,
    )
    recheck_count =  0
    updated_res = None
    while True:
        # needs_recheck_res = llm.get_answer(question=if_loop_prompt, model = 'gpt-4o')
        # needs_recheck = "yes" in needs_recheck_res.strip().lower()
        prompt = (
            "Please determine whether the following structure context sufficiently addresses the question. "
            "Return 'true' if it does, otherwise return 'false', and provide a brief explanation.\n"
            f"Question: {question}\n"
            f"Context: {current_table}\n"
            "Please return 'true' or 'false' followed by an explanation."
        )

        response = llm.get_answer(question=prompt, model='gpt-4o')
        # print(response)
        if "true" in response.lower():
            needs_recheck = False
        else:
            needs_recheck = True

        if not needs_recheck:
            break 

        recheck_count +=1
        if recheck_count > MAX_RECHECK_ATTEMPTS:
            break
        
        print(f"Table needs recheck against original data. Re-analyzing...{recheck_count}")
        # Re-analyze the original data with the current table as context
        recheck_prompt = PROMPTS["TABLE_RECHECK_PROMPT"].format(
            current_table=current_table,
            # query=question,
            content=data,
            tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
            record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
            completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],

        )   
        # print(recheck_prompt)
        # updated_res = llm.get_answer(question=recheck_prompt, model='deepseek-r1')

        updated_res = to_table(data, question)
        updated_res['need_recheck'] = needs_recheck
        # print(updated_res)
        
        current_table = updated_res['answer']

    return updated_res, needs_recheck


class TableMerger:
    """
    Merge multiple tables into a single table.
    """
    def __init__(self):
        self.tables = []
    def _strip_angle_brackets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean all string fields, remove angle brackets <>
        """
        def strip_cell(x):
            if isinstance(x, str):
                return x.strip().strip("<>").strip('"').strip("'")
            return x
        # 清洗数据列的内容
        df = df.apply(lambda col: col.apply(strip_cell))

        # 清洗列名中的尖括号
        df.columns = df.columns.str.strip('<>').str.strip('"').str.strip("'")

        return df
    
    def add_table(self, process_tables: str, PROMPTS: dict):
        """
        Add a DataFrame table generated by parsing the structured string
        """
        title, header, rows, description = parse_table(
            process_tables, 
            PROMPTS["DEFAULT_TUPLE_DELIMITER"],
            PROMPTS["DEFAULT_RECORD_DELIMITER"],
            PROMPTS["DEFAULT_COMPLETION_DELIMITER"]
        )
        df = pd.DataFrame(rows, columns=header)
        df = df.replace('', np.nan)
        df = df.infer_objects(copy=False)
        df = self._strip_angle_brackets(df)
        self.tables.append(df)

    def _natural_join(self, left: pd.DataFrame, right: pd.DataFrame, how="outer") -> pd.DataFrame:
        common_cols = list(set(left.columns).intersection(set(right.columns)))
        if not common_cols:
            raise ValueError(f"❌ Cannot execute natural join, no common columns: \nleft={left.columns}\nright={right.columns}")

        # 🚨 确保公共列的类型一致，统一为字符串（特别适用于 Company、Year 等文本型）
        for col in common_cols:
            left[col] = left[col].astype(str)
            right[col] = right[col].astype(str)

        merged = pd.merge(left, right, on=common_cols, how=how, suffixes=('', '_dup'))

        for col in merged.columns:
            if col.endswith('_dup'):
                base = col[:-4]
                merged[base] = merged[base].combine_first(merged[col])
                merged.drop(columns=[col], inplace=True)
        return merged


    def merge(self, join_type="outer") -> pd.DataFrame:
        """
        Execute all table natural join merge, support outer / inner join
        """
        assert join_type in ["outer", "inner"], "join_type must be 'outer' or 'inner'"

        if not self.tables:
            raise ValueError("⚠️ No tables added")

        from functools import reduce
        # return reduce(lambda l, r: self._natural_join(l, r, how=join_type), self.tables)
    
        try:
            merged = reduce(lambda l, r: self._natural_join(l, r, how=join_type), self.tables)
            return merged
        except Exception as e:
            print(f"❌ Merge failed, reason: {e}")
            return pd.DataFrame()
    
    def union(self) -> pd.DataFrame:
        """
        Execute all table union operation (concatenate rows), require all tables have the same column name and column order.
        """
        if not self.tables:
            raise ValueError("⚠️ No tables added")

        # 确保所有表格的列名一致
        columns = self.tables[0].columns
        for df in self.tables[1:]:
            # 如果列名不一致，进行统一列名操作（可以选择填充或调整列名）
            if not all(df.columns == columns):
                df = df[columns]  # 调整为统一的列顺序
                # 如果列名不同，则添加 NaN 列
                missing_cols = set(columns) - set(df.columns)
                for col in missing_cols:
                    df[col] = np.nan
                df = df[columns]  # 确保列的顺序一致

        # 使用 concat 来进行并集（union）操作
        return pd.concat(self.tables, ignore_index=True)


    
def generate_tables(chunks, schema):
    merger = TableMerger()
    print(len(chunks))
    for chunk in chunks:
        table = to_table(chunk, schema)
        structured = table["answer"]
        merger.add_table(structured, PROMPTS)
    join_results = merger.merge(join_type="outer")
    union_results = merger.union()

    return join_results, union_results


async def main():
    model = "llama-3.1-8b-instruct"
    entity_type = {  # 你的表结构定义
        "Name": "Person's Name",
        "Age": "Person's Age",
        "City": "Person's City"
    }
    
    # 多个data（每条是原始文本）
    datas = [
        "John is 25 years old and lives in New York.",
        "Alice, aged 30, resides in Los Angeles.",
        "Bob is from Chicago and he is 40."
    ]

    # 并发执行
    tasks = [to_table(data, entity_type, model=model) for data in datas]
    results = await asyncio.gather(*tasks)

    # 查看结果
    for i, res in enumerate(results):
        print(f"\n==== Result {i+1} ====")
        print(res["structured_data"])  # 打印Markdown表格
        print(f"Latency: {res['latency']:.2f} seconds")


if __name__ == "__main__":
    # 正式运行
    asyncio.run(main())