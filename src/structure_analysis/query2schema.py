"""
This module provides functionality for extracting a schema from a query.

The module includes the following functions:
- parse_schema: Parse the schema.
- get_schema: Extracts a schema from a query.
"""

import llm
import re
import os
import json

from src.prompt import PROMPTS
import llm.global_config as config


def parse_schema(schema):
    """
    Parse the schema from the response.

    Args:
        schema (str): The schema from the response.

    Returns:
        str: The cleaned schema.
    """
    cleaned_text = re.sub(r"<think>.*?</think>", "", schema, flags=re.DOTALL).strip()
    match = re.search(r"\s*(\(.*?\))", cleaned_text) 
    if match:
        return match.group(1)
    else:
        return cleaned_text
    
def get_schema(query):
    """
    Extracts a schema from a query.

    Args:
        query (str): The query to extract the schema from.

    Returns:
        str: The schema.
    """
    examples = PROMPTS['SCHEMA_CONSTRUCTION_EXAMPLES']
    raw_prompt = PROMPTS['SCHEMA_CONSTRUCTION']
    schema_construction_prompt = raw_prompt.format(
        examples=examples,
        query=query,
    )
    output = llm.get_answer(question = schema_construction_prompt, model='gpt-4o')
    return parse_schema(output)



if __name__ == '__main__':
    model = "gpt-4o" #gpt, llama, deepseek
    config.set_model(model)

    question = "How did the net profit margin of <Tesla> compare to its operating costs in 2022?"
    schema = get_schema(question)
    print(schema)