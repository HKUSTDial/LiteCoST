"""
This module provides functionality for converting data into a graph structure.
"""
import llm as llm
import ast
import os, logging, re
import math, jieba, pickle

from src.rm_prompt import RM_PROMPTS
from src.prompt import PROMPTS
from src.extract.graph import Graph
from src.structure_analysis.query2schema import get_schema
from src.seek.main import split
from src.utils import token_length, extract_answer_content, extract_intermediate_results
from src.extract.graph import Graph
from typing import List, Tuple, Optional

import time

ENCODER = None
jieba.setLogLevel(log_level=logging.INFO)



def parse_triples(response, tuple_delimiter, record_delimiter, completion_delimiter):
    """
    Parse the triples from the response.

    Args:
        response (str): The response from the LLM.
        tuple_delimiter (str): The delimiter for the tuples.
        record_delimiter (str): The delimiter for the records.
        completion_delimiter (str): The delimiter for the completion.

    Returns:
        list: A list of triples.
    """
    # 保存中间结果
    # steps = extract_intermediate_results(response)
    
    # if steps:
    #     print(steps)
    #     entries = steps[-1].split(completion_delimiter)[0].split(record_delimiter)
    # else:
    #     entries = response.split(completion_delimiter)[0].split(record_delimiter)
    
    entries = response.split(completion_delimiter)[0].split(record_delimiter)
    
    entities = []
    relationships = []
    descriptions = []

    # 处理每个条目
    for entry in entries:
        # 如果包含"entity"则处理为实体
        if "entity" in entry:
            parts = entry.replace('\n', '').split(tuple_delimiter)
            entities.append(parts[1:4])
            # descriptions.append(parts[-1].split(record_delimiter)[0][:-1])
        # 如果包含"relationship"则处理为关系
        elif "relationship" in entry:
            parts = entry.replace('\n', '').split(tuple_delimiter)
            relationships.append(parts[1:4])
            # if "complete" in entry: descriptions.append(parts[-1].split(completion_delimiter)[0][:-1])
            # else: descriptions.append(parts[-1].split(record_delimiter)[0][:-1])
    
    # return relationships, steps
    return relationships


def to_graph(data: str, entity_type: str) -> dict:
    """
    Convert the input data into a graph structure.

    Args:
        data (str): The input data to be converted.
        entity_type (str): The entity type to be used in the graph construction.

    Returns:
        dict: A dictionary containing the graph structure.
    """
    print(entity_type)

    # Construct the LLM input
    examples = PROMPTS['GRAPH_CONSTRUCTION_EXAMPLES']
    raw_prompt = RM_PROMPTS['GRAPH_CONSTRUCTION_R1']
    # entity_type = [item.strip().strip("'") for item in entity_type.strip("()").split(',')]
    triple_extraction_prompt = raw_prompt.format(
        # examples=examples,
        schema=entity_type,
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        # query=question,
        content=data,
    )

    start_time = time.time()
    triples = llm.get_answer(question=triple_extraction_prompt, model='yizhan')
    latency = time.time() - start_time
    print(triples)

    # 清理中间思考过程    
    if "<think>" in triples:
        process_triples = re.sub(r"<think>.*?</think>", "", triples, flags=re.DOTALL).strip()
    else:
        process_triples = triples

    process_triples = extract_answer_content(process_triples)

    try:
        refined_data = parse_triples(
            process_triples, 
            PROMPTS["DEFAULT_TUPLE_DELIMITER"], 
            PROMPTS["DEFAULT_RECORD_DELIMITER"], 
            PROMPTS["DEFAULT_COMPLETION_DELIMITER"]
        )
    except Exception as e:
        print("parse_triples failed:", e)
        refined_data = triples
        # steps = triples
    

    print(refined_data)
    print("----")

    # return {"schema": entity_type, "structure": refined_data, "steps": steps, 'cot': triples}
    return {"schema": entity_type, "structured_data": refined_data, 'cot': triples, 'cot_length': token_length(triples), 'answer': process_triples, 'latency': latency} 

class GraphMerger(Graph):
    """
    A class for merging multiple graphs into a single graph.
    """
    def __init__(self, description="Merged Graph"):
        super().__init__(description=description)

    def add_graph(self, triplets: List[List[str]]) -> None:
        """
        Add a small graph (constructed from a list of triplets) to the main graph.

        Args:
            triplets (List of triplet list): The triplets of the small graph, e.g. [["A", "knows", "B"], ["B", "likes", "C"]]
        """
        temp_graph = Graph()
        temp_graph.create_graph_from_triplets(triplets)
        self.add_subgraph(temp_graph)

def generate_graphs(chunks, schema):
    """
    Generate a graph from a list of chunks.

    Args:
        chunks (List of str): The chunks to be merged into a single graph.
        schema (str): The schema of the graph.
    """
    print(len(chunks))

    merger = GraphMerger()
    for chunk in chunks:
        graph = to_graph(chunk, schema)
        triplets = graph["structured_data"]

        merger.add_graph(triplets)

    return merger.get_edges()

if __name__ == "__main__":
    question = "What is Amazon's FY2017 days payable outstanding (DPO)? DPO is defined as: 365 * (average accounts payable between FY2016 and FY2017) \/ (FY2017 COGS + change in inventory between FY2016 and FY2017). Round your answer to two decimal places. Address the question by using the line items and information shown within the balance sheet and the P&L statement."
    context = "Table of Contents\nAMAZON.COM, INC.\nCONSOLIDATED STATEMENTS OF OPERATIONS\n(in millions, except per share data)\n \n \nYear Ended December 31,\n \n2015\n \n2016\n \n2017\nNet product sales\n$\n79,268 $\n94,665 $\n118,573\nNet service sales\n27,738 \n41,322 \n59,293\nTotal net sales\n107,006 \n135,987 \n177,866\nOperating expenses:\n \n \n \nCost of sales\n71,651 \n88,265 \n111,934\nFulfillment\n13,410 \n17,619 \n25,249\nMarketing\n5,254 \n7,233 \n10,069\nTechnology and content\n12,540 \n16,085 \n22,620\nGeneral and administrative\n1,747 \n2,432 \n3,674\nOther operating expense, net\n171 \n167 \n214\nTotal operating expenses\n104,773 \n131,801 \n173,760\nOperating income\n2,233 \n4,186 \n4,106\nInterest income\n50 \n100 \n202\nInterest expense\n(459) \n(484) \n(848)\nOther income (expense), net\n(256) \n90 \n346\nTotal non-operating income (expense)\n(665) \n(294) \n(300)\nIncome before income taxes\n1,568 \n3,892 \n3,806\nProvision for income taxes\n(950) \n(1,425) \n(769)\nEquity-method investment activity, net of tax\n(22) \n(96) \n(4)\nNet income\n$\n596 $\n2,371 $\n3,033\nBasic earnings per share\n$\n1.28 $\n5.01 $\n6.32\nDiluted earnings per share\n$\n1.25 $\n4.90 $\n6.15\nWeighted-average shares used in computation of earnings per share:\n \n \n \nBasic\n467 \n474 \n480\nDiluted\n477 \n484 \n493\nSee accompanying notes to consolidated financial statements.\n38\nTable of Contents\nAMAZON.COM, INC.\nCONSOLIDATED BALANCE SHEETS\n(in millions, except per share data)\n \n \nDecember 31,\n \n2016\n \n2017\nASSETS\n \n \nCurrent assets:\n \n \nCash and cash equivalents\n$\n19,334 $\n20,522\nMarketable securities\n6,647 \n10,464\nInventories\n11,461 \n16,047\nAccounts receivable, net and other\n8,339 \n13,164\nTotal current assets\n45,781 \n60,197\nProperty and equipment, net\n29,114 \n48,866\nGoodwill\n3,784 \n13,350\nOther assets\n4,723 \n8,897\nTotal assets\n$\n83,402 $\n131,310\nLIABILITIES AND STOCKHOLDERS EQUITY\n \n \nCurrent liabilities:\n \n \nAccounts payable\n$\n25,309 $\n34,616\nAccrued expenses and other\n13,739 \n18,170\nUnearned revenue\n4,768 \n5,097\nTotal current liabilities\n43,816 \n57,883\nLong-term debt\n7,694 \n24,743\nOther long-term liabilities\n12,607 \n20,975\nCommitments and contingencies (Note 7)\n \nStockholders equity:\n \n \nPreferred stock, $0.01 par value:\n \n \nAuthorized shares 500\n \n \nIssued and outstanding shares none\n \n\nCommon stock, $0.01 par value:\n \n \nAuthorized shares 5,000\n \n \nIssued shares 500 and 507\n \n \nOutstanding shares 477 and 484\n5 \n5\nTreasury stock, at cost\n(1,837) \n(1,837)\nAdditional paid-in capital\n17,186 \n21,389\nAccumulated other comprehensive loss\n(985) \n(484)\nRetained earnings\n4,916 \n8,636\nTotal stockholders equity\n19,285 \n27,709\nTotal liabilities and stockholders equity\n$\n83,402 $\n131,310\nSee accompanying notes to consolidated financial statements.\n40"
    std_answer = "93.86"
    schema = get_schema(question)
    entity_type = [item.strip().strip("'") for item in schema.strip("()").split(',')]

    # result = to_graph(context, entity_type)
    # print(result)

    chunks = split(context)
    graph = generate_graphs(chunks, entity_type)
    print(graph)
