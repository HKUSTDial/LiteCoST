"""
This module provides functionality for selecting the most suitable data structure for a given question.

The module includes the following functions:
- parse_response: Parses the response from the LLM to extract the answer and reason.
- select: Based on the question, selects the most suitable data structure from the given options.
"""

import re, json
from typing import Tuple, Any, Union

import llm as llm


def parse_response(json_str):
    """
    Parse the response from the LLM to extract the answer and reason.

    Args:
        json_str (str): The response from the LLM.

    Returns:
        dict: A dictionary containing the answer and reason.
    """
    try:
        if '```json' in json_str:
            json_str = re.search(r'```json\n(.*?)```', json_str, re.DOTALL).group(1).strip()
        else:
            json_str = json_str.strip()
        
        data = json.loads(json_str)
        return {
            "ans": data.get("answer", "N/A"),
            "reason": data.get("reason", "No reason provided")
        }
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format"}

# def select(question: str, structures=None, need_explain = False) -> tuple[str | None, Any] | str | None:
def select(question: str, structures=None, need_explain=False) -> Union[tuple[Union[str, None], Any], Union[str, None]]:
    """
    Based on the question, select the most suitable data structure from the given options.

    Args:
        question (str): The user's question.
        structures (list): A list of available structures (e.g., ["Text description", "Table", "Graph", "Tree"]).

    Returns:
        str: The most suitable data structure chosen by the LLM.
    """
    if structures is None:
        # structures = ["Table", "Graph", "Tree", "Text Description"]
        structures = ["Table", "Graph", "Text Description"]

    prompt = (
        "This is a data structure selection task. Based on the given `question`, choose the most suitable data structure "
        "to answer the question. You can choose from the following options:\n")
    for structure in structures:
        prompt += f"- {structure}"
        if structure == "Text Description": prompt += " (simple facts, single-step queries)\n"
        if structure == "Table": prompt += " (statistical comparisons, multi-source data)\n"
        # if structure == "Tree": prompt += " (hierarchical relationships, classifications)\n"
        if structure == "Graph": prompt += " (entity connections, network analysis)\n"
    prompt += (
        "Return your answer in format: {\"answer\": \"data structure\", \"reason\": \"concise_explanation\"}\n\n"
        f"The question is: {question}\n\n"
    )
    prompt += (
        "Guidelines:\n"
        "1. If the question requires aggregating and comparing numbers/attributes from multiple sources -> Use Table\n"
        "2. If the question focuses on connections between entities -> Use Graph\n"
        # "3. If the question involves parent-child relationships or taxonomies -> Use Tree\n"
        "4. If the question can be answered with direct text extraction -> Use Text Description"
    )

    ans = llm.get_answer(prompt, model='gpt-4o')

    result = parse_response(ans)
    ds = result["ans"]
    
    # print(result)

    if need_explain:
        return ds, ans

    return ds

if __name__ == '__main__':
    ans = select("Under other key management personnel of the group, how many executive directors are included?", need_explain=True)
    print(ans)