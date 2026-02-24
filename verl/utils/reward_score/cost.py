"""
1. format reward
Must satisfy the format:
<think></think> and <ans></ans>
where the thinking process is step-by-step
<step1></step1>...<step3></step3>
answer must be successfully parsed automatically: parse_table, parse_graph

If correct, r = 0.5; else, r = 0

2. answer reward
The content inside <answer></answer> is scored by the large model/whether it can answer the query
If yes/higher than a threshold: r = 1
else, r = -1
"""
import json
import os
from openai import OpenAI
from typing import Dict, List, Optional

import re
import time
from verl.utils.reward_score.litecost.prompt import PROMPTS
from verl.utils.reward_score.litecost import gpt as llm
from verl.utils.reward_score.litecost.utils import *


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def sanitize_generated_answer(raw: str, missing_placeholder="N/A") -> str:
    """
    Clean model outputs, fix empty column markers <|><|>, and normalize format.
    
    Args:
        raw: raw model output string
        missing_placeholder: placeholder for missing fields, default "N/A"
        
    Returns:
        cleaned string, ready for parse()
    """
    # Remove trailing marker
    cleaned = raw.replace("<|COMPLETE|>", "").strip()

    # Fix consecutive <|><|> → <|>missing<|>
    # Replace any consecutive empty <|> markers with <|>N/A<|>
    while "<|><|>" in cleaned:
        cleaned = cleaned.replace("<|><|>", f"<|>{missing_placeholder}<|>")

    # Replace leading/trailing empty fields
    cleaned = re.sub(r'<\|>\s*(?=\)\s*$)', f'<|>{missing_placeholder}', cleaned)

    # Remove redundant spaces/newlines
    cleaned = re.sub(r'\n+', '', cleaned).strip()

    return cleaned


def parse_table(response, tuple_delimiter, record_delimiter, completion_delimiter):
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
    entries = response.split(completion_delimiter)[0].split(record_delimiter)
    
    entities = []
    relationships = []

    # 处理每个条目
    for entry in entries:
        # 如果包含"entity"则处理为实体
        if "entity" in entry:
            parts = entry.replace('\n', '').split(tuple_delimiter)
            entities.append(parts[1:4])
        # 如果包含"relationship"则处理为关系
        elif "relationship" in entry:
            parts = entry.replace('\n', '').split(tuple_delimiter)
            relationships.append(parts[1:4])
    return relationships

class StructuredDataParser:
    def parse(self, response: str) -> dict:
        """Parse two structured formats (entity-relationship / table)."""
        # Preprocess
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

        for record in records:
            # Remove parentheses and split
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
        """Validate structural correctness of parsed data."""
        # Check table consistency
        for table in data["tables"]:
            if table["headers"] and table["rows"]:
                header_count = len(table["headers"])
                for row in table["rows"]:
                    if len(row) != header_count:
                        raise ValueError(
                            f"Table row length mismatch: header has {header_count}, row has {len(row)}"
                        )
        
        # Check relationship references
        entity_names = {e["name"] for e in data["entities"]}
        for rel in data["relationships"]:
            if rel["from"] not in entity_names:
                raise ValueError(f"Relationship refers to undefined entity: {rel['from']}")
            if rel["to"] not in entity_names and not self._is_numeric_value(rel["to"]):
                raise ValueError(f"Relationship refers to undefined entity or non-numeric: {rel['to']}")

        
class StructuredDataEvaluator:
    def __init__(self, api_key, base_url):
        # os.environ["OPENAI_API_BASE"] = base_url
        # os.environ["OPENAI_API_KEY"] = api_key
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.parser = StructuredDataParser()  # 基于之前代码的解析器
        
        
    def evaluate(self, generated_answer: str, ground_truth: str) -> float:
        """核心评估流程"""
        # 步骤1：数据解析
        try:
            gt_data = self.parser.parse(ground_truth)
            gen_data = self.parser.parse(generated_answer)
        except Exception as e:
            print(f"解析失败: {str(e)}") 
            return 0.0
        
        # print(gt_data)
        # print(gen_data)
        # 步骤2：结构验证
        structure_score = self._validate_structure(gt_data, gen_data)

        # 步骤3：语义验证
        semantic_score = self._gpt_semantic_compare(gt_data, gen_data)

        # 步骤4：综合评分
        # if abs(structure_score) < 1e-10:
        #     final_score = 0.0
        # else:
        #     final_score = structure_score * 0.3 + semantic_score * 0.7
        final_score = structure_score * 0.3 + semantic_score * 0.7

        # print(f"结构匹配得分: {structure_score}, 语义匹配得分: {semantic_score}, 总得分: {final_score}")
        
        # return {
        #     "score": final_score,
        #     "structure_score": structure_score,
        #     "semantic_score": semantic_score,
        #     "details": self._generate_diff_report(gt_data, gen_data)
        # }
        return final_score
    def _validate_structure(self, gt: dict, gen: dict) -> float:
        """结构相似性评分"""
        score = 0.0
        
        # 图结构检查
        if gt['entities'] or gt['relationships']:
            # 实体匹配检查
            gt_entities = {(e['name'].lower(), e['type'].lower()) for e in gt['entities']}
            gen_entities = {(e['name'].lower(), e['type'].lower()) for e in gen['entities']}
            entity_match = len(gt_entities & gen_entities) / max(1, len(gt_entities))
            # 关系三元组检查
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
        
        # 表格结构检查
        if gt['tables'] and gen['tables']:
            gt_table = gt['tables'][0]
            gen_table = gen['tables'][0]
            header_match = set(gt_table['headers']) == set(gen_table['headers'])
            row_count_match = len(gt_table['rows']) == len(gen_table['rows'])
            
            score = (0.2 * header_match + 0.8 * row_count_match) * 100
            
        return score

    # 换成BGE看看效果
    def _gpt_semantic_compare(self, gt: dict, gen: dict) -> float:
        """基于GPT-4o的语义对比"""
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
        - 若【生成数据】为空或无实质内容（如"暂无"、"空白"），请直接返回 score = 0，并在 warnings 中注明"生成数据为空"。

        2. **核心字段覆盖情况**  
        - 检查【生成数据】是否覆盖了【真实数据】中的核心字段（如公司名、营收、时间点等）。  
        - 核心字段大量缺失将显著影响得分；非关键字段缺失影响较小。  
        - 字段命名不一致但语义一致（如"营收" vs "营业收入"）可视为等价。

        3. **字段间语义对齐质量**  
        请关注以下语义维度，判断对应字段是否等价表达了同一内容：  
        - 数值表达是否一致（如"1,000 万"与"1000万"）  
        - 单位转换是否合理，汇率/比例/时间跨度是否明确  
        - 实体指称是否一致（如"微软公司" vs "Microsoft"）  
        - 时间表达是否表示相同的时间点或周期  
        - 事实关系是否一致（如"创始人是马云" vs "马云创办该公司"）

        4. **语义等价判断（不降分但需识别）**  
        以下情形可视为语义一致，请勿因此降低得分：  
        - 合理的数字标准化（如：1,000,000 vs 100 万）  
        - 合理简称（如：Google vs GOOG）  
        - 时间格式差异（如：2023Q4 vs 2023-10-01）  
        - 同义替换（如："营收" vs "营业收入"）

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
                    model="gpt-4o-mini",
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
                # print(result)
                return float(result.get("score", 0.0))  
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:  # 如果不是最后一次重试，则等待
                    time.sleep(1)
        print("Max retries reached, returning 0.0")
        return 0.0

    def _format_for_gpt(self, data: dict) -> str:
        """将结构化数据转换为自然语言描述"""
        output = []
        
        # 实体部分
        if data['entities']:
            output.append("关键实体：")
            for e in data['entities']:
                output.append(f"- {e['name']} ({e['type']})")
        
        # 关系部分
        if data['relationships']:
            output.append("\n实体关系：")
            for r in data['relationships']:
                output.append(f"- {r['from']} → [{r['relation']}] → {r['to']}")
        
        # 表格部分
        for table in data['tables']:
            output.append(f"\n表格：{table['title']}")
            output.append("| " + " | ".join(table['headers']) + " |")
            for row in table['rows']:
                output.append("| " + " | ".join(row) + " |")
        
        return "\n".join(output)

    def _generate_diff_report(self, gt: dict, gen: dict) -> dict:
        """生成结构化差异报告"""
        report = {
            "missing_entities": [],
            "extra_entities": [],
            "value_discrepancies": [],
            "relation_errors": []
        }
        
        # 实体对比
        gt_entities = {(e['name'], e['type']) for e in gt['entities']}
        gen_entities = {(e['name'], e['type']) for e in gen['entities']}
        report["missing_entities"] = list(gt_entities - gen_entities)
        report["extra_entities"] = list(gen_entities - gt_entities)
        
        # 数值对比（示例）
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
        """数值标准化（示例）"""
        try:
            return float(re.sub(r'[^\d.]', '', value))
        except:
            return 0.0



def process_steps(text):
    pattern = r'(Step \d+:.*?)(?=Step \d+:|\Z)'
    matches = re.findall(pattern, text, re.DOTALL)
    
    # 如果找不到，尝试匹配 "1. ", "2. " 结构
    if not matches:
        pattern = r'(\d+\.\s+.*?)(?=\d+\.\s+|\Z)'
        matches = re.findall(pattern, text, re.DOTALL)

    return matches
def score_step1(step1, truth_step1, schema):
    # result1 = extract_schema_entities_llm(step1, schema)
    # result2 = extract_schema_entities_llm(truth_step1, schema)
    # print(f"step1: {result1}")
    # print(f"truth_step1: {result2}")
    # print("--------------------------------")
    # return compare_dicts(result1, result2) 
    return compare_extracted_entities(schema, step1, truth_step1)

def score_step2(step2, truth_step2, schema):
    # result1 = convert_entities_to_pairs_llm(step2, schema)
    # result2 = convert_entities_to_pairs_llm(truth_step2, schema)
    # print(f"step2: {result1}")
    # print(f"truth_step2: {result2}")
    # print("--------------------------------")
    # return compare_n_tuples(result1, result2)
    return compare_n_tuples_llm(schema, step2, truth_step2)


def score_step3(step3, truth_step3): 
    import markdown
    def compare_markdown_html(md1: str, md2: str) -> bool:
        html1 = markdown.markdown(md1)
        html2 = markdown.markdown(md2)
        # 也可以进一步去除空白
        return html1.strip() == html2.strip()

    return 0.0

# 添加全局训练步数计数器
_training_step = 0
_max_training_steps = 300  # 与 GRPO 的 max_steps 保持一致

def update_training_step(step: int):
    """更新训练步数"""
    global _training_step
    _training_step = step

def calculate_lambda(answer_correct: bool, format_correct: bool) -> float:
    """
    计算过程奖励的权重系数 λ
    
    参数:
        answer_correct: 答案是否正确
        format_correct: 格式是否正确
    
    返回:
        float: λ 值
    """
    if not format_correct:
        return 1.0  # 如果格式错误，λ = 1
    elif answer_correct:
        return 1.5  # 如果答案正确，λ > 1
    else:
        return 0.5  # 如果答案错误，λ < 1
    
def calculate_dynamic_lambda(answer_correct: bool, format_correct: bool) -> float:
    """
    计算动态的过程奖励权重系数 λ
    
    参数:
        answer_correct: 答案是否正确
        format_correct: 格式是否正确
    
    返回:
        float: λ 值
    """
    global _training_step
    
    # 计算训练进度比例 (0 到 1)
    progress = min(_training_step / _max_training_steps, 1.0)
    
    # 基础 lambda 值
    if not format_correct:
        base_lambda = 1.0  # 如果格式错误，λ = 1
    elif answer_correct:
        base_lambda = 1.5  # 如果答案正确，λ > 1
    else:
        base_lambda = 0.5  # 如果答案错误，λ < 1
    
    # 根据训练进度调整 lambda
    if answer_correct:
        # 随着训练进行，正确答案的奖励逐渐增加
        adjusted_lambda = base_lambda + (0.5 * progress)
    else:
        # 随着训练进行，错误答案的惩罚逐渐增加
        adjusted_lambda = base_lambda - (0.2 * progress)
    
    return adjusted_lambda

# 添加一个全局字典用于缓存答案评分
_answer_score_cache = {}

def process_reward(solution_str, ground_truth, extra_info):
    """计算过程奖励分数。

    Args:
        solution_str: 解决方案文本
        ground_truth: 标准答案

    Returns:
        float: 过程奖励分数
    """
    score = 0.0

    # 检查格式是否正确
    format_correct = False        
    format_reward_score = format_score(solution_str)

    if format_reward_score == 1.0:
        format_correct = True

    # 检查答案是否正确
    answer_correct = False
    try:
        answer_content = extract_xml_answer(solution_str)
        gt_content = extract_xml_answer(ground_truth)
        answer_score = evaluator.evaluate(answer_content, gt_content)
        # 将评分存入缓存
        cache_key = f"{answer_content}_{gt_content}"
        _answer_score_cache[cache_key] = answer_score
        
        if answer_score == 100.0:
            answer_correct = True
    except Exception:
        answer_correct = False

    # 计算 λ 值
    lambda_value = calculate_lambda(answer_correct, format_correct)

    gt = extra_info["answer"]
    reasoning_gt = re.search(r'<reasoning>(.*?)</reasoning>', gt, re.DOTALL)
    reasoning_gt = reasoning_gt.group(1).strip() if reasoning_gt else gt
    truth_steps = process_steps(reasoning_gt)
    # print(f"length of truth_steps: {len(truth_steps)}")

    
    if not truth_steps or len(truth_steps) < 3:
        return 0.5  # gt有问题，直接给0.5

    # 1. simple format
    pattern = r"^<reasoning>[\s\S]*?</reasoning>\s*<answer>[\s\S]*?</answer>$"
    match1 = re.match(pattern, solution_str)
    if match1:         
        # 2. step format
        reasoning_content = re.search(r'<reasoning>(.*?)</reasoning>', solution_str, re.DOTALL)
        reasoning = reasoning_content.group(1).strip() if reasoning_content else solution_str

        match2 = process_steps(reasoning)
        if match2:  
            # 1. 提取每个step内容
            if len(match2) > 0:
                steps = []
                for i in range(len(match2)):
                    steps.append(match2[i])
                step1 = steps[0] if len(steps) > 0 else None 
                step2 = steps[1] if len(steps) > 1 else None
                step3 = steps[2] if len(steps) > 2 else None
            else:
                return 0.0
            
            # 2. 针对每个step打分
            if step1:
                schema = extra_info["schema"]
                # print(f"schema: {schema}")
                if score_step1(step1, truth_steps[0], schema):
                    score += 1
            if step2:
                if score_step2(step2, truth_steps[1], schema):
                    score += 1

            # 应用 λ 调整
            adjusted_score = (float(score) / 2) * lambda_value
            # adjusted_score = (float(score) / 2) 
            return adjusted_score

        else:
            return 0.0
    else:
        return 0.0

def answer_reward(solution_str, ground_truth):
    """计算答案奖励分数。

    Args:
        solution_str: 解决方案文本
        ground_truth: 标准答案

    Returns:
        float: 答案奖励分数
    """
    answer_content = extract_xml_answer(solution_str)
    gt_content = extract_xml_answer(ground_truth)
    
    # 检查缓存中是否有评分
    cache_key = f"{answer_content}_{gt_content}"
    if cache_key in _answer_score_cache:
        score = _answer_score_cache[cache_key]
    else:
        score = evaluator.evaluate(answer_content, gt_content)
        time.sleep(1)

    min_score = 0   # 实际可能的最小值
    max_score = 100 # 实际可能的最大值
    normalized_score = (score - min_score) / (max_score - min_score)

    return float(normalized_score)

def format_score(response):
    score = 0.0
    match_count = 0

    # 1. simple format
    pattern = r"^<reasoning>[\s\S]*?</reasoning>\s*<answer>[\s\S]*?</answer>$"
    match1 = re.match(pattern, response)
    if match1: 
        # score += 0.40
        match_count += 1
    
    # 2. step format  
    reasoning_content = re.search(r'<reasoning>(.*?)</reasoning>', response, re.DOTALL)
    reasoning = reasoning_content.group(1).strip() if reasoning_content else response

    pattern2 = r'(Step \d+:.*?)(?=Step \d+:|\Z)'
    match2 = re.findall(pattern2, reasoning, re.DOTALL)

    # 如果找不到，尝试匹配 "1. ", "2. " 结构
    pattern22 = r'(\d+\.\s+.*?)(?=\d+\.\s+|\Z)'
    match22 = re.findall(pattern22, reasoning, re.DOTALL)

    if match2 or match22:  
        # score += 0.30
        match_count += 1

    # 3. answer format
    answer_content = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    answer = answer_content.group(1).strip() if answer_content else response
    # print(answer)
    # print("-------")
    
    answer_match = False
    if "table" in answer:
        try:
            title, header, rows, description = parse_table(answer, PROMPTS["DEFAULT_TUPLE_DELIMITER"], PROMPTS["DEFAULT_RECORD_DELIMITER"], PROMPTS["DEFAULT_COMPLETION_DELIMITER"])
            # score += 0.30  # 解析成功，增加分数
            answer_match = True
        except Exception:
            pass  # 解析失败，不加分
    if "entity" in answer:
        try:
            graph = parse_graph(answer, PROMPTS["DEFAULT_TUPLE_DELIMITER"], PROMPTS["DEFAULT_RECORD_DELIMITER"], PROMPTS["DEFAULT_COMPLETION_DELIMITER"])
            # score += 0.30  # 解析成功，增加分数
            answer_match = True
        except Exception:
            pass
    
    if answer_match:
        match_count += 1

    # 只有三个match都正确时才返回1.0分
    if match_count == 3:
        score = 1.0

    return score
    
def format_reward(completions, **kwargs):
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
        score = format_score(response)

        # if added_score:
        #     score = 1.0
        print(f"format reward: {score}")
        rewards.append(score)
    
    return rewards

evaluator = StructuredDataEvaluator(api_key="", base_url="")

def compute_score(solution_str, ground_truth, extra_info, answer_weight=1.0, format_weight=1.0, process_weight=1.0):
    """计算综合得分，包含答案奖励、格式奖励和过程奖励。

    Args:
        solution_str: 解决方案文本
        ground_truth: 标准答案
        answer_weight: 答案奖励权重，默认为1.0
        format_weight: 格式奖励权重，默认为1.0
        process_weight: 过程奖励权重，默认为1.0

    Returns:
        float: 综合得分
    """
    # 计算格式得分
    format_scores = format_score(solution_str)
    
    # 计算过程得分
    process_scores = process_reward(solution_str, ground_truth, extra_info)

    # 计算过程得分
    answer_scores = answer_reward(solution_str, ground_truth)

    # 分别打印三个score的分数
    # print(f"格式得分（format_scores）: {format_scores}")
    # print(f"过程得分（process_scores）: {process_scores}")
    # print(f"答案得分（answer_scores）: {answer_scores}")
    # 计算加权总分
    final_score = answer_weight * answer_scores + format_weight * format_scores + process_weight * process_scores
    # final_score = answer_weight * answer_scores + format_weight * format_scores
    # final_score = format_weight * format_scores + process_weight * process_scores

    return final_score

if __name__ == '__main__':
    # total_rewards = get_reward(question, response)
    # print(total_rewards)

    # format_rewards = format_reward(completions=completions)
    # print(format_rewards)
    # answer_rewards = answer_reward(prompts=prompts, completions=completions, ground_truth=ground_truth)
    # print(answer_rewards)


    # 初始化评估器
    evaluator = StructuredDataEvaluator(api_key="", base_url="")

    # 测试数据
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

    # similarity_score = compute_similarity(ground_truth, generated)
    # print(f"Semantic similarity score: {similarity_score}")