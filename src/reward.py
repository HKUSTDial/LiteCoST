import json
import os
from openai import OpenAI
from typing import Dict, List, Optional

import llm as llm
import re
import time
from .prompt import PROMPTS


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def sanitize_generated_answer(raw: str, missing_placeholder="N/A") -> str:
    """
    Clean model output, fix empty column markers <|><|>, and perform format cleaning.
    
    Args:
        raw: Raw model output string
        missing_placeholder: Placeholder for empty fields, defaults to "N/A"
        
    Returns:
        Cleaned string that can be directly passed to parse()
    """
    # Remove trailing markers
    cleaned = raw.replace("<|COMPLETE|>", "").strip()

    # Fix all consecutive <|><|> → <|>missing<|>
    # Replace all consecutive <|> → <|>N/A<|>
    while "<|><|>" in cleaned:
        cleaned = cleaned.replace("<|><|>", f"<|>{missing_placeholder}<|>")

    # Replace empty columns at the beginning or end, e.g. <|>xxx<|> → xxx<|>N/A
    cleaned = re.sub(r'<\|>\s*(?=\)\s*$)', f'<|>{missing_placeholder}', cleaned)

    # Clean excessive spaces and newlines
    cleaned = re.sub(r'\n+', '', cleaned).strip()

    return cleaned

def parse_table(response, tuple_delimiter, record_delimiter, completion_delimiter):
    """
    Parse table data from model output

    Args:
        response: Model output string
        tuple_delimiter: Delimiter for tuple elements
        record_delimiter: Delimiter for records
        completion_delimiter: Delimiter for completion
    Returns:
        title: Title of the table
        description: Description of the table
        header: List of headers
        rows: List of rows
    """

    def strip_trailing_parenthesis(part):
        return part.rstrip(')")')
    
    entries = response.split(completion_delimiter)[0].split(record_delimiter)
    title = ""
    description = ""
    header = []
    rows = []
    
    for entry in entries:
        parts = entry.replace('\n', '').split(tuple_delimiter)
        if not parts:
            continue
        if "table" in parts[0]:
            title = parts[1]
            description = parts[2]
        elif "header" in parts[0]:
            header = [strip_trailing_parenthesis(part.strip('"'))  for part in parts[1:]]
        elif "row" in parts[0]:
            rows.append([strip_trailing_parenthesis(part.strip('"'))  for part in parts[1:]])
    return title, header, rows, description

def parse_graph(response, tuple_delimiter, record_delimiter, completion_delimiter):
    """
    Parse graph data from model output

    Args:
        response: Model output string
        tuple_delimiter: Delimiter for tuple elements
        record_delimiter: Delimiter for records
        completion_delimiter: Delimiter for completion
    Returns:
        relationships: List of relationships
    """

    entries = response.split(completion_delimiter)[0].split(record_delimiter)
    
    entities = []
    relationships = []

    # Process each entry
    for entry in entries:
        # If it contains "entity", process it as an entity
        if "entity" in entry:
            parts = entry.replace('\n', '').split(tuple_delimiter)
            entities.append(parts[1:4])
        # If it contains "relationship", process it as a relationship
        elif "relationship" in entry:
            parts = entry.replace('\n', '').split(tuple_delimiter)
            relationships.append(parts[1:4])
    return relationships

class StructuredDataParser:
    def parse(self, response: str) -> dict:
        """Parse two types of structured data (entity relationships/tables)"""
        # Preprocessing
        cleaned = sanitize_generated_answer(response)

        if not cleaned:
            raise ValueError("Input data is empty")
        
        # Result container
        result = {
            "entities": [],
            "relationships": [],
            "tables": []
        }
        
        # Split records
        records = [r.strip() for r in cleaned.split('##') if r.strip()]
        current_table = []
        print(records)

        for record in records:
            # Extract record content (remove outer parentheses)
            content = re.sub(r'^\(|\)$', '', record)
            parts = [re.sub(r'^"|"$', '', p.strip()) for p in content.split('<|>') if p.strip()]
            
            if not parts:
                continue
                
            record_type = parts[0].lower()
            
            # Entity record
            if record_type == "entity":
                if len(parts) >= 3:
                    result["entities"].append({
                        "name": parts[1],
                        "type": parts[2],
                        "value": parts[3] if len(parts) > 3 else None
                    })
            
            # Relationship record
            elif record_type == "relationship":
                if len(parts) >= 4:
                    result["relationships"].append({
                        "from": parts[1],
                        "relation": parts[2],
                        "to": parts[3]
                    })
            
            # Table record
            elif record_type == "table":
                current_table = {
                    "title": parts[1] if len(parts) > 1 else "",
                    "description": parts[2] if len(parts) > 2 else "",
                    "headers": [],
                    "rows": []
                }
                result["tables"].append(current_table)
            
            # Header record
            elif record_type == "header" and current_table:
                current_table["headers"] = parts[1:]
            
            # Row record
            elif record_type == "row" and current_table:
                current_table["rows"].append(parts[1:])
        
        self._validate_result(result)
        return result
    
    def _validate_result(self, data: dict):
        """Data validity validation"""
        # Check table structure consistency
        for table in data["tables"]:
            if table["headers"] and table["rows"]:
                header_count = len(table["headers"])
                for row in table["rows"]:
                    if len(row) != header_count:
                        raise ValueError(f"Table row data column number does not match, the table header has {header_count} columns, the actual row has {len(row)} columns")
        
        # Check relationship reference validity
        entity_names = {e["name"] for e in data["entities"]}
        for rel in data["relationships"]:
            if rel["from"] not in entity_names:
                raise ValueError(f"Relationship references undefined entity: {rel['from']}")
            if rel["to"] not in entity_names and not self._is_numeric_value(rel["to"]):
                raise ValueError(f"Relationship references undefined entity or non-numeric value: {rel['to']}")
    
    def _is_numeric_value(self, s: str) -> bool:
        """Check if the string is a numeric type"""
        try:
            float(re.sub(r'[^\d.-]', '', s))
            return True
        except ValueError:
            return False
        
class StructuredDataEvaluator:
    def __init__(self, api_key, base_url):
        # os.environ["OPENAI_API_BASE"] = base_url
        # os.environ["OPENAI_API_KEY"] = api_key
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.parser = StructuredDataParser()  # 基于之前代码的解析器
        
        
    def evaluate(self, generated_answer: str, ground_truth: str) -> float:
        """Core evaluation process"""
        # Step 1: Data parsing
        try:
            gt_data = self.parser.parse(ground_truth)
            gen_data = self.parser.parse(generated_answer)
        except Exception as e:
            print(f"Parsing failed: {str(e)}") 
            return 0.0
        
        print(gt_data)
        print(gen_data)
        # Step 2: Structure validation
        structure_score = self._validate_structure(gt_data, gen_data)

        # Step 3: Semantic validation
        semantic_score = self._gpt_semantic_compare(gt_data, gen_data)

        # Step 4: Comprehensive scoring
        final_score = structure_score * 0.3 + semantic_score * 0.7

        print(f"Structure matching score: {structure_score}, Semantic matching score: {semantic_score}, Total score: {final_score}")
        
        # return {
        #     "score": final_score,
        #     "structure_score": structure_score,
        #     "semantic_score": semantic_score,
        #     "details": self._generate_diff_report(gt_data, gen_data)
        # }
        return final_score
    def _validate_structure(self, gt: dict, gen: dict) -> float:
        """Structure similarity scoring"""
        score = 0.0
        
        # Graph structure check
        if gt['entities'] or gt['relationships']:
            # Entity matching check
            gt_entities = {(e['name'].lower(), e['type'].lower()) for e in gt['entities']}
            gen_entities = {(e['name'].lower(), e['type'].lower()) for e in gen['entities']}
            entity_match = len(gt_entities & gen_entities) / max(1, len(gt_entities))
            # Relationship triple check
            gt_relations = {
                (r['from'].lower(), r['relation'].lower(), r['to'].lower()) 
                for r in gt['relationships']
            }
            gen_relations = {
                (r['from'].lower(), r['relation'].lower(), r['to'].lower())
                for r in gen['relationships']
            }

            relation_match = (
                len(gt_relations & gen_relations) / max(1, len(gt_relations))
            )
            score = (entity_match * 0.6 + relation_match * 0.4) * 100

            # print(entity_match)
            # print(relation_match)
            # print(score)
        
        # Table structure check
        if gt['tables'] and gen['tables']:
            gt_table = gt['tables'][0]
            gen_table = gen['tables'][0]
            header_match = set(gt_table['headers']) == set(gen_table['headers'])
            row_count_match = len(gt_table['rows']) == len(gen_table['rows'])
            
            score = (0.2 * header_match + 0.8 * row_count_match) * 100
            
        return score

    def _gpt_semantic_compare(self, gt: dict, gen: dict) -> float:
        """
        Semantic comparison based on GPT-4o
        """
        prompt1 = f"""【语义验证任务】
        请对比以下两组结构化数据的语义一致性，重点关注：
        1. 数值的等价性（如单位转换、格式差异）
        2. 实体指称的一致性（如公司全称与简称）
        3. 时间表达的准确性
        4. 关系表述的语义等价性

        【真实数据】
        {self._format_for_gpt(gt)}

        【生成数据】
        {self._format_for_gpt(gen)}

        请按以下JSON格式返回结果：
        {{
            "score": 0-100,
            "key_differences": [
                "差异描述1",
                "差异描述2" 
            ],
            "warnings": [
                "潜在问题提示1",
                "潜在问题提示2"
            ]
        }}"""
        
        prompt = f"""【语义验证任务说明】
        请你担任一名严谨的语义一致性评估器，负责判断【生成数据】与【真实数据】在信息内容与语义上的一致程度。请从以下四个维度逐步评估其一致性，并最终给出一个 0 到 100 的整型评分（score），同时提供主要差异点（key_differences）和潜在问题提示（warnings）。

        ---

        【评估步骤】

        1. **空值检测**  
        - 若【生成数据】为空或无实质内容（如“暂无”、“空白”），请直接返回 score = 0，并在 warnings 中注明“生成数据为空”。

        2. **核心字段覆盖情况**  
        - 检查【生成数据】是否覆盖了【真实数据】中的核心字段（如公司名、营收、时间点等）。  
        - 核心字段大量缺失将显著影响得分；非关键字段缺失影响较小。  
        - 字段命名不一致但语义一致（如“营收” vs “营业收入”）可视为等价。

        3. **字段间语义对齐质量**  
        请关注以下语义维度，判断对应字段是否等价表达了同一内容：  
        - 数值表达是否一致（如“1,000 万”与“1000万”）  
        - 单位转换是否合理，汇率/比例/时间跨度是否明确  
        - 实体指称是否一致（如“微软公司” vs “Microsoft”）  
        - 时间表达是否表示相同的时间点或周期  
        - 事实关系是否一致（如“创始人是马云” vs “马云创办该公司”）

        4. **语义等价判断（不降分但需识别）**  
        以下情形可视为语义一致，请勿因此降低得分：  
        - 合理的数字标准化（如：1,000,000 vs 100 万）  
        - 合理简称（如：Google vs GOOG）  
        - 时间格式差异（如：2023Q4 vs 2023-10-01）  
        - 同义替换（如：“营收” vs “营业收入”）

        ---

        【真实数据】
        {self._format_for_gpt(gt)}

        【生成数据】
        {self._format_for_gpt(gen)}

        ---

        【输出格式】
        请严格按照如下 JSON 结构输出：

        {{
        "score": 0-100,
        "key_differences": [
            "差异描述1",
            "差异描述2",
            ...
        ],
        "warnings": [
            "潜在问题提示1",
            "潜在问题提示2",
            ...
        ]
        }}

        注意事项：
        - 请勿输出多余解释，只输出 JSON 对象；
        - 若生成数据为空，请将 score 设置为 0；
        - 若没有明显差异或问题，key_differences 和 warnings 可为 ["无"]。
        """

        # print(prompt)
        
        system_prompt = "你是专业的数据质量分析师，擅长发现结构化数据中的语义差异"
        # response = llm.get_answer(text=prompt, system_prompt=system_prompt)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{
                        "role": "system",
                        "content": system_prompt
                    }, {
                        "role": "user",
                        "content": prompt
                    }],
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )
                
                result = json.loads(response.choices[0].message.content)
                print(result)
                return float(result.get("score", 0.0))  
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:  # 如果不是最后一次重试，则等待
                    time.sleep(1)
        print("Max retries reached, returning 0.0")
        return 0.0

    def _format_for_gpt(self, data: dict) -> str:
        """Convert structured data to natural language description"""
        output = []
        
        # Entity part
        if data['entities']:
            output.append("Key entities:")
            for e in data['entities']:
                output.append(f"- {e['name']} ({e['type']})")
        
        # Relationship part
        if data['relationships']:
            output.append("\nEntity relationships:")
            for r in data['relationships']:
                output.append(f"- {r['from']} → [{r['relation']}] → {r['to']}")
        
        # Table part
        for table in data['tables']:
            output.append(f"\nTable: {table['title']}")
            output.append("| " + " | ".join(table['headers']) + " |")
            for row in table['rows']:
                output.append("| " + " | ".join(row) + " |")
        
        return "\n".join(output)

    def _generate_diff_report(self, gt: dict, gen: dict) -> dict:
        """Generate structured difference report"""
        report = {
            "missing_entities": [],
            "extra_entities": [],
            "value_discrepancies": [],
            "relation_errors": []
        }
        
        # Entity comparison
        gt_entities = {(e['name'], e['type']) for e in gt['entities']}
        gen_entities = {(e['name'], e['type']) for e in gen['entities']}
        report["missing_entities"] = list(gt_entities - gen_entities)
        report["extra_entities"] = list(gen_entities - gt_entities)
        
        # Numerical comparison (example)
        for gt_e, gen_e in zip(gt['entities'], gen['entities']):
            if gt_e['type'] == 'Domestic Sales Growth' and gen_e['type'] == 'Domestic Sales Growth':
                if self._normalize_value(gt_e['name']) != self._normalize_value(gen_e['name']):
                    report["value_discrepancies"].append({
                        "field": "国内销售增长率",
                        "ground_truth": gt_e['name'],
                        "generated": gen_e['name']
                    })
        
        return report

    def _normalize_value(self, value: str) -> float:
        """Normalize value (example)"""
        try:
            return float(re.sub(r'[^\d.]', '', value))
        except:
            return 0.0
        

def format_reward(completions, **kwargs):
    """
    Format reward
    Args:
        completions: list of completions
    Returns:
        list of rewards
    """
    rewards = []
    # responses = [completion[0]["content"] for completion in completions]
    # responses = completions

    responses = []
    for completion in completions:
        if isinstance(completion, list) and isinstance(completion[0], dict):
            responses.append(completion[0]["content"])
        elif isinstance(completion, str):
            responses.append(completion)
        else:
            raise ValueError(f"Unexpected completion format: {type(completion)} -> {completion}")

    for response in responses:
        score = 0.0
        added_score = False

        # print(response)
        # print("------")
        # 1. simple format
        pattern = r"^<reasoning>[\s\S]*?</reasoning>\s*<answer>[\s\S]*?</answer>$"
        match1 = re.match(pattern, response)
        if match1: 
            score += 0.40
            added_score = True
        
        # 2. step format
        reasoning_content = re.search(r'<reasoning>(.*?)</reasoning>', response, re.DOTALL)
        reasoning = reasoning_content.group(1).strip() if reasoning_content else response

        pattern = r'(Step \d+:.*?)(?=Step \d+:|\Z)'
        match2 = re.findall(pattern, reasoning, re.DOTALL)

        if match2 and len(match2) == 3:  
            score += 0.30
            added_score = True

        # 3. answer format
        answer_content = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        answer = answer_content.group(1).strip() if answer_content else response
        # print(answer)
        # print("-------")
        if "table" in answer:
            try:
                title, header, rows, description = parse_table(answer, PROMPTS["DEFAULT_TUPLE_DELIMITER"], PROMPTS["DEFAULT_RECORD_DELIMITER"], PROMPTS["DEFAULT_COMPLETION_DELIMITER"])
                score += 0.30  # Parse successfully, add score
                added_score = True
            except Exception:
                pass  # Parse failed, no score added
        if "entity" in answer:
            try:
                graph = parse_graph(answer, PROMPTS["DEFAULT_TUPLE_DELIMITER"], PROMPTS["DEFAULT_RECORD_DELIMITER"], PROMPTS["DEFAULT_COMPLETION_DELIMITER"])
                score += 0.30  # Parse successfully, add score
                added_score = True
            except Exception:
                pass  # Parse failed, no score added

        # if added_score:
        #     score = 1.0
        print(f"format reward: {score}")
        rewards.append(score)
    
    return rewards

evaluator = StructuredDataEvaluator(api_key="", base_url="")
def answer_reward(prompts, completions, answer, **kwargs):
    """
    Answer reward
    Args:
        prompts: list of prompts
        completions: list of completions
        answer: list of answers
    Returns:
        list of rewards
    """
    # responses = [completion[0]['content'] for completion in completions]
    # responses = completions
    responses = []
    for completion in completions:
        if isinstance(completion, list) and isinstance(completion[0], dict):
            responses.append(completion[0]["content"])
        elif isinstance(completion, str):
            responses.append(completion)
        else:
            raise ValueError(f"Unexpected completion format: {type(completion)} -> {completion}")
        
    rewards = []
    score = 0.0

    for response, gt in zip(responses, answer):
        answer_content = extract_xml_answer(response)
        score = evaluator.evaluate(answer_content, gt)
        time.sleep(1)

        min_score = 0   # Replace with actual minimum value
        max_score = 100 # Replace with actual maximum value
        normalized_score = (score - min_score) / (max_score - min_score)

        # print(normalizded_score)
        rewards.append(float(normalized_score))

        print(f"generated answer: {answer_content}")
        print(f"answer reward: {score}")
        
    # Log completion lengths
    completion_lengths = [len(response.split()) for response in responses]
    return rewards
    

if __name__ == '__main__':


    # Initialize evaluator
    evaluator = StructuredDataEvaluator(api_key="", base_url="")

    # Test data
    ground_truth = '''
    ("entity"<|>"Johnson & Johnson"<|>"Company")##
    ("entity"<|>"3.1%"<|>"Domestic Sales Growth")##
    ("entity"<|>"(0.6)%"<|>"International Sales Growth")##
    ("relationship"<|>"Johnson & Johnson"<|>"Domestic Growth"<|>"3.1%")<|COMPLETE|>
    '''

    generated = '''
    ("entity"<|>"J&J"<|>"Company")##
    ("entity"<|>"3.1%"<|>"Domestic Sales Growth")##
    ("relationship"<|>"J&J"<|>"Domestic Growth"<|>"3.1%")<|COMPLETE|>
    '''

    # ground_truth = """
    # (\"table\"<|> \"Equity Awards and Performance Milestones for 2012\" <|> \"This table summarizes equity awards, performance milestones, and compensation expense for equity awards for the fiscal year 2012.\")##\n(\"header\"<|> \"Year\" <|> \"Equity Awards\" <|> \"Performance Milestones\" <|> \"Equity Award Compensation Expense\")##\n(\"row\"<|> \"2012\" <|> \"Restricted stock and restricted stock units granted\" <|> \"Achievement of prescribed performance targets\" <|> \"$3.3 million\")##\n<|COMPLETE|>
    # """
    # generated = """
    # (\"table\"<|>\"Stock-Based Compensation for 2012\"<|>\"Source: ABIOMED, Inc.\")##\n(\"header\"<|>\"Year\"<|>\"Equity Awards\"<|>\"Performance Milestones\"<|>\"Equity Award Compensation Expense\")##\n(\"row\"<|>\"2012\"<|>\"$3.3 million\"<|>\"Achieved for 184,000 shares and probable for 100,000 shares\"<|>\"$3.6 million\")##\n<|COMPLETE|>
    # """

    # 执行评估
    result = evaluator.evaluate(generated, ground_truth)
    print(json.dumps(result, indent=2))