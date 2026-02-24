import re
import ast
import json
import random
from openai import OpenAI

import tiktoken
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

random.seed(10086)

api_key = ""
api_url = ""

def token_length(text):
    return len(encoding.encode(text, disallowed_special=()))



def extract_n_tuples(text):
        """
        从文本中提取元组关系，处理元素中可能包含逗号的情况
        Args:
            text: 包含元组关系的文本
        Returns:
            list: 提取出的元组列表
        """
        tuples = []
        try:
            # 使用正则表达式匹配括号内的元组
            pattern = r'\((.*?)\)'
            matches = re.findall(pattern, text)
            
            for match in matches:
                # 使用状态机解析元组元素,处理元素中的逗号
                elements = []
                current_element = []
                in_quotes = False
                quote_char = None
                in_backtick = False
                in_number = False
                
                for char in match:
                    if char in ['"', "'"]:
                        if not in_quotes:
                            in_quotes = True
                            quote_char = char
                        elif char == quote_char:
                            in_quotes = False
                    elif char == '`':
                        in_backtick = not in_backtick
                    elif char.isdigit() or char in ['$', '¥', '.']:
                        in_number = True
                        current_element.append(char)
                    elif char == ',' and (in_quotes or in_backtick or in_number):
                        current_element.append(char)
                    elif char == ',' and not in_quotes and not in_backtick and not in_number:
                        if current_element:
                            elements.append(''.join(current_element).strip())
                            current_element = []
                        in_number = False
                    else:
                        if not char.isspace():
                            in_number = False
                        current_element.append(char)
                
                if current_element:  # 添加最后一个元素
                    elements.append(''.join(current_element).strip())
                    
                # 移除空元素
                elements = [e for e in elements if e]
                if len(elements) == 1:  # 如果只有一个元素，添加一个空元组
                    tuples.append((elements[0],))
                elif len(elements) > 1:  # 多个元素创建普通元组
                    tuples.append(tuple(elements))
            
            return tuples
        except Exception as e:
            print(f"解析元组时出错: {e}")
            return []

def normalize_elem(elem):
    """
    归一化元组元素：全部转小写，去除首尾引号，去除所有空格。
    如果包含数字，去除货币符号、逗号、空格、引号。
    """
    if isinstance(elem, str):
        elem = elem.lower().strip()
        elem = elem.replace('[inferred]', '')
        elem = elem.strip('\'"')  # 去除首尾引号
        elem = elem.replace(' ', '')  # 去除所有空格
        if re.search(r'\d', elem):
            # 去除货币符号、逗号、空格、引号
            elem = re.sub(r'[\$,¥,\'\"\\s]', '', elem)
    return elem


def is_english(text):
    # 判断文本是否主要为英文
    english_chars = re.findall(r'[a-zA-Z]', text)
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    return len(english_chars) > len(chinese_chars)


def extract_schema_entities_llm(text, schema):
    """
    从文本中提取指定schema对应的实体
    
    Args:
        text (str): 待提取的文本
        schema (list): schema列表,如 ["Year", "Beginning Balance"]
        
    Returns:
        dict: 提取的实体字典,
    """
    api_key = "sk-pkHdpajDryeAqj5v4e29011eDa3440Ae942dFf22F188521a"
    api_url = "https://svip.yi-zhan.top/v1/"
    client = OpenAI(api_key=api_key, base_url=api_url)
    
    # 根据文本语言构建提示词
    if is_english(text):
        prompt = (
            f"Please extract entities for the following schemas from the text: {schema}\n"
            "For each schema, extract all matching entities and return them in a list.\n"
            "Return the result in a Python dictionary format like:\n"
            "{\"schema1\": [entity1, entity2, ...], \"schema2\": [entity1, ...]}.\n\n"
            f"Text:\n{text}"
        )
    else:
        prompt = (
            f"请从下面的文本中提取以下schema对应的实体: {schema}\n"
            "对每个schema,提取所有匹配的实体并放入列表中。\n"
            "返回字典形式如:\n"
            "{\"schema1\": [entity1, entity2, ...], \"schema2\": [entity1, ...]}。\n"
            f"文本如下：\n{text}"
        )

    max_retries = 3
    for attempt in range(max_retries):
        # 调用API提取实体
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt} if attempt == 0
                else {
                    "role": "user", "content": prompt,
                    "role": "assistant", "content": content,
                    "role": "user", "content": "请重新生成一个有效的JSON格式响应"
                }
            ],
            temperature=0.0
        )
        content = response.choices[0].message.content
        # print(prompt)
        # print(content)
        # 解析返回的JSON内容
        try:
            if '```json' in content:
                content = re.search(r'```json\n(.*?)```', content, re.DOTALL).group(1).strip()
            if '```python' in content:
                content = re.search(r'```python\n(.*?)```', content, re.DOTALL).group(1).strip()
        
            result = json.loads(content)
            return result  # 如果成功解析就直接返回
        except:
            if attempt == max_retries - 1:  # 最后一次尝试失败
                return content  # 所有重试都失败后返回空字典
            continue  # 否则继续下一次尝试

def compare_dicts(dict1: dict, dict2: dict) -> bool:
    """
    比较两个字典是否完全相同，所有字符串归一化处理都用normalize_elem函数
    Args:
        dict1: 第一个字典
        dict2: 第二个字典
    Returns:
        bool: 两个字典是否完全相同
    """
    # 检查键是否相同
    if set(dict1.keys()) != set(dict2.keys()):
        return False
    
    # 检查每个键对应的值是否相同
    for key in dict1:
        v1 = dict1[key]
        v2 = dict2[key]
        # 如果值是列表,需要转换为集合来比较,忽略顺序
        if isinstance(v1, list) and isinstance(v2, list):
            # 确保所有元素都是字符串
            set1 = set(normalize_elem(str(item)) for item in v1)
            set2 = set(normalize_elem(str(item)) for item in v2)
            # if set1 != set2:
            #     return False
            if not set2.issubset(set1):
                return False
            
        # 如果值是字符串，归一化后比较
        elif isinstance(v1, str) and isinstance(v2, str):
            if normalize_elem(v1) != normalize_elem(v2):
                return False
        # 其他情况直接比较
        elif v1 != v2:
            return False
    return True

def compare_extracted_entities(schema: list, generated_text: str, ground_truth_text: str) -> bool:
    """
    比较两个文本提取的实体是否相同，允许单位、空格等差异
    
    Args:
        schema: schema列表，如 ["Year", "Net Revenue"]
        generated_text: 生成的文本
        ground_truth_text: 真实文本
    
    Returns:
        bool: 提取的实体是否相同
    """
    client = OpenAI(api_key=api_key, base_url=api_url)


    # 构建提示词
    prompt = f"""
    Please extract entities for the following schemas from the two texts and compare if they are equivalent.
    
    Schema: {schema}
    
    Generated Text:
    {generated_text}
    
    Ground Truth Text:
    {ground_truth_text}
    
    Requirements:
    1. Extract entities for each schema from both texts
    2. Compare if the extracted entities are equivalent, ignoring:
       - Units (e.g., $, ¥, %)
       - Spaces
       - Case differences
       - Commas in numbers
    3. Return a Json dictionary with three keys:
       - "entities1": extracted entities from Generated Text
       - "entities2": extracted entities from Ground Truth Text
       - "is_equivalent": True/False
    
    Example:
    If Text 1 has "Net Revenue: $1,234" and Text 2 has "Net Revenue: 1234",
    they should be considered equivalent.

    IMPORTANT: Return ONLY the JSON dictionary, no other text or explanation.
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # 调用大模型
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            response = response.choices[0].message.content
            try:
                # 解析响应
                # print(f"response:", response)

                # 检查response是否为空
                if not response or response.strip() == "":
                    print("Response为空，重新生成...")
                    return compare_extracted_entities(schema, generated_text, ground_truth_text)

                if '```json' in response:
                    response = re.search(r'```json\n(.*?)```', response, re.DOTALL).group(1).strip()
                if '```python' in response:
                    response = re.search(r'```python\n(.*?)```', response, re.DOTALL).group(1).strip()
                
                response = response.replace("'", '"')
                response = response.replace("True", "true").replace("False", "false")

                result = json.loads(response)
                return result.get("is_equivalent", False)
            except:
                if "true" in response or "True" in response:
                    return True
                else:
                    return False
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:  # 最后一次尝试
                print("All attempts failed")
                return False
            continue  # 继续下一次尝试
    return False    
    
    

def convert_entities_to_pairs_llm(text: str, schema: list) -> list:
    """
    使用大模型将文本转换为n-tuples列表
    
    Args:
        entities_dict: 实体字典（作为参考）
        text: 相关文本内容
        language: 提示词语言，'en' 或 'zh'
    
    Returns:
        list: n-tuples列表，格式如 [(e1,e2,e3,...)]
    """
    client = OpenAI(api_key=api_key, base_url=api_url)
    if is_english(text):
        # 中英文提示词
        prompt = f"""
        Please convert the following text into n-tuples based on the schema.

        Schema:
        {schema}

        Text:
        {text}
        
        Requirements:
        1. Extract key information from the text
        2. Convert them into n-tuples format: (e1,e2,e3,...)
        3. Only return the n-tuples, no other explanations
        4. Each tuple should contain all relevant elements
        5. Format example: (revenue, 2015, company_name)
        """
    else:
        prompt = f"""
        请根据schema将以下文本转换为n-tuples格式。
        
        Schema:
        {schema}
        
        文本内容：
        {text}
        
        要求：
        1. 从文本中提取关键信息
        2. 将它们转换为n-tuples格式：(e1,e2,e3,...)
        3. 只返回n-tuples，不要其他解释
        4. 每个元组应包含所有相关元素
        5. 格式示例：(收入, 2015, 公司名称)
        """
    
    # 调用大模型
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    content = response.choices[0].message.content
    # print(content)
    # 解析响应，提取n-tuples
    tuple_pattern = r'\((.*?)\)'
    matches = re.findall(tuple_pattern, content)
    
    if matches:
        # 提取所有n-tuples
        n_tuples = []
        for match in matches:
            values = [v.strip() for v in match.split(',')]
            n_tuples.append(tuple(values))
        return n_tuples
    else:
        print("无法从大模型响应中提取n-tuples")
        return []

def compare_n_tuples(list1, list2):
    """
    判断两组 n-tuples 列表是否一样（无序全等，全部小写，数字字符串归一化）
    """
    def norm_tuple(t):
        return tuple(normalize_elem(e) for e in t)
    set1 = set(norm_tuple(t) for t in list1)
    set2 = set(norm_tuple(t) for t in list2)
    return set1 == set2


def compare_n_tuples_llm(schema: list, generated_text: str, ground_truth_text: str) -> bool:
    """
    比较两个文本转换的n-tuples是否相同，允许单位、空格等差异
    
    Args:
        schema: schema列表，如 ["Year", "Net Revenue"]
        generated_text: 生成的文本
        ground_truth_text: 真实文本
    
    Returns:
        bool: 转换的n-tuples是否相同
    """
    client = OpenAI(api_key=api_key, base_url=api_url)
    
    # 构建提示词
    prompt = f"""
    Please convert the following two texts into n-tuples based on the schema and compare if they are equivalent.

    Schema:
    {schema}

    Generated Text:
    {generated_text}

    Ground Truth Text:
    {ground_truth_text}
    
    Requirements:
    1. Convert both texts into n-tuples format: (e1,e2,e3,...)
    2. Compare if the n-tuples are equivalent, ignoring:
       - Units (e.g., $, ¥, %)
       - Spaces
       - Case differences
       - Commas in numbers
    3. Return a **Json dictionary** with three keys:
       - "tuples1": n-tuples from Generated Text
       - "tuples2": n-tuples from Ground Truth Text
       - "is_equivalent": True/False
    
    Example:
    If Text 1 has "Net Revenue: $1,234 in 2023" and Text 2 has "Net Revenue: 1234 in 2023",
    they should be considered equivalent.

    IMPORTANT: Return ONLY the JSON dictionary, no other text or explanation.
    """
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # 调用大模型
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            response = response.choices[0].message.content
            try:
                # print(f"response:", response)
                # 检查response是否为空
                if not response or response.strip() == "":
                    print("Response为空，重新生成...")
                    return compare_n_tuples_llm(schema, generated_text, ground_truth_text)
                    
                if '```json' in response:
                    response = re.search(r'```json\n(.*?)```', response, re.DOTALL).group(1).strip()
                if '```python' in response:
                    response = re.search(r'```python\n(.*?)```', response, re.DOTALL).group(1).strip()
                if '```' in response:
                    response = re.search(r'```\n(.*?)```', response, re.DOTALL).group(1).strip()
                
                response = response.replace("'", '"')
                response = response.replace("True", "true").replace("False", "false")

                
                result = json.loads(response)
                return result.get("is_equivalent", False)
            except:
                if "true" in response or "True" in response:
                    return True
                else:
                    return False
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:  # 最后一次尝试
                print("All attempts failed")
                return False
            continue  # 继续下一次尝试
    return False  # 所有尝试都失败



# 示例用法
if __name__ == "__main__":
    text_en = """
    Step 1: **Entity Extraction**  
    Relevant entities extracted based on schema:  
    - `Year` → Identified years: 2015, 2016, 2017  
    - `Beginning Balance` → Corresponding values: `$85207`, `$96838`, `$98966`  
    """
    text_zh = """
    第一步：**实体抽取**
    根据schema抽取到的相关实体：
    - `年份` → 识别年份：2015年，2016年，2017年
    - `期初余额` → 对应数值：`¥85207`，`¥96838`，`¥98966`
    """

    text_en2 = """
    Step 1: **Entity Extraction**  
    Relevant entities extracted based on schema:  
    - `Year` → Identified years: 2015, 2016, 2017, 2018  
    - `Beginning Balance` → Corresponding values: `$85207`, `96,838 billion`, `$98,966`, `100 billion`  
    """

    text_en22 = """
    Step 1: **Entity Extraction**  
    Relevant entities extracted based on schema:  
    - `Year` → Identified years: 2015, 2016, 2017 
    - `Beginning Balance` → Corresponding values: `$85,207`, `$96,838`, `$98966`  
    """

    text_en3 = """
        Step 1: **Entity Extraction**  \nRelevant entities extracted based on schema:  \n- `Year` → Identified years: 2015, 2016, 2017  \n- `Beginning Balance` → Corresponding values: `$85207`, `$96838`, `$98966`  \n\nIntermediate Result:  \nExtracted Entities:  \n- Year: [2015, 2016, 2017]  \n- Beginning Balance: [$85207, $96838, $98966]\n\n---\n\nStep 2: **Entity Linking**  \nLinking raw entities to schema columns using exact keyword matching:  \n- From the table:\n  - 2015 → Beginning Balance: `$85207`  \n  - 2016 → Beginning Balance: `$96838`  \n  - 2017 → Beginning Balance: `$98966`  \n\nIntermediate Result:  \nEntity Relationships:  \n- (2015, `$85207`)  \n- (2016, `$96838`)  \n- (2017, `$98966`)\n\n---\n\nStep 3: **Summarization**  \nOrganize linked entities into a structured table format based on the schema:  \n\nIntermediate Result:  \nDraft Table:  \n| Year | Beginning Balance |  \n|------|-------------------|  \n| 2015 | $85207           |  \n| 2016 | $96838           |  \n| 2017 | $98966           |  \n\n---\n\nStep 4: **Data Validation**  \nCheck extracted data for accuracy and adherence to schema:  \n- Check formatting: Dates and dollar amounts are retained as in raw content.  \n- Missing data: n/a (all required entities were present).  \n- No changes or updates needed during validation.  \n\nIntermediate Result:  \nValidated Table:  \n| Year | Beginning Balance |  \n|------|-------------------|  \n| 2015 | $85207           |  \n| 2016 | $96838           |  \n| 2017 | $98966           |  \n\n---\n\nStep 5: **Final Output Formatting**  \nPrepare final output in specified structure: 
    """

    schema = ["Year", "Beginning Balance"]
    # result1 = extract_schema_entities_llm(text_en, schema)
    # result2 = extract_schema_entities_llm(text_en2, schema)

    result1 = {'Net Revenue': ['$1095.9', '$998.7'], 'Year': ['2002', '2003']}
    result2 = {'Net Revenue': ['$ 1095.9', '$ 998.7'], 'Year': ['2002', '2003']}
    result3 = {'Capital Expenditures': ['$447', '$476', '$617'], 'Year': [2016, 2017, 2018]}
    result4 = {'Capital Expenditures': ['$ 447', '$ 476', '$ 617'], 'Year': ['2016', '2017', '2018']}
    # print(compare_dicts(result1, result2))
    # print(compare_dicts(result3, result4))

    print(compare_extracted_entities(schema, text_en2, text_en22))

    step2_text = """
    Step 2: **Entity Linking**  \nLinking raw entities to schema columns using exact keyword matching:  \n- From the table:\n  - 2015 → Beginning Balance: `$85207`  \n  - 2016 → Beginning Balance: `$96838`  \n  - 2017 → Beginning Balance: `$98966`  \n\nIntermediate Result:  \nEntity Relationships:  \n- (2015, $85,207)  \n- (2016, `$96838 billion`)  \n- (2017, `$98966`)\n\n---\n\n
    """
    step2_text2 = """
    Step 2: **Entity Linking**  \nLinking raw entities to schema columns using exact keyword matching:  \n- From the table:\n  - 2015 → Beginning Balance: `$85207`  \n  - 2016 → Beginning Balance: `$96838`  \n  - 2017 → Beginning Balance: `$98966`  \n\nIntermediate Result:  \nEntity Relationships:  \n- (2015, `$85207`)  \n- (2016, `$96838`)  \n- (2017, `$98966`)\n\n---\n\n
    """
    # result1 = extract_n_tuples(step2_text)
    # result2 = extract_n_tuples(step2_text2)
    # print(result1)
    # print(result2)
    # print(compare_n_tuples(result1, result2))
    print(compare_n_tuples_llm(schema, step2_text, step2_text2))

    # pairs_en = convert_entities_to_pairs_llm(step2_text2, schema)
    # print("English prompt result:", pairs_en)  # 输出: [('revenue', '2015', 'entergy arkansas')]

    result1 = [("'$1095.9'", '2002'), ("'$998.7'", '2003')]
    result2 = [("'$ 1095.9'", "'2002'"), ("'$ 998.7'", "'2003'")]
    # print(compare_n_tuples(result1, result2))

