# schema+cot
PROMPTS = {}

PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

PROMPTS['STRUCTURE_DECISION'] = """

Instruction: 
You are a Data Structure Selection Agent specializing in matching problem types to optimal data representations. Analyze the given question's intent and select the most suitable structure from: Text, Tree, Table, Graph.

Decision Criteriaon:
1. Text Description: For simple problems, such as single-hop queries, fact retrieval, or scenarios where structured data is unnecessary, the optimal strategy is to present the information in the document in the form of a textual description.

2. Table: For statistical questions, such as analyzing and comparing data across multiple documents or data sources (e.g., financial reports, sales records), the optimal strategy is to present the information in the form of a table.

3. Tree: For questions with hierarchical or recursive properties, such as organizational structures, classification systems, or pathfinding problems, the optimal strategy is to present the information in the document in the form of a tree structure.

4. Graph: For relational reasoning questions that require modeling interconnected entities (e.g., social networks, knowledge graphs, or system dependency mappings), the optimal strategy is to present the information in the form of a graph structure.

### Examples:

Query: {query}


"""
instruction_mapper = {
    "Table": {
        "EE": "Entity Extraction: Identify relevant entities from raw content based on the schema, and ensure extracted entities align with schema column definitions. Finally generate an intermediate result listing all extracted entities.",
        "EL": "Entity Linking: Map values to schema columns using exact keyword matching, directly link the raw entities as they appear, without performing any additional inference. Finally generate an intermediate result that outputs the relationships between entities (in the form of tuples), explaining how each entity is connected.",
        "Sum": "Summarization: Based on the entities and linking relationships from the previous steps, summarize the information and organize it into a table structured format that conforms to the Schema. Generate a draft table as an intermediate result, ensuring that each Schema column is correctly populated."
    },
    "Graph": {
        "EE": "Entity Extraction: Identify all entities from the raw text that match the Schema definition, with the following type: entity_name (Name of the entity) and entity_type (One of the Schema types). Finally format each entity as (\"entity\"<|><entity_name><|><entity_type>).",
        "EL": "Entity Linking: Identify all triples of (head, relation, tail) that are *clearly related* to each other from the entities extracted, with each triple representing a relationship. Finally format each relationship as (\"relationship\"{tuple_delimiter}<head>{tuple_delimiter}<relation>{tuple_delimiter}<tail>), explaining how each entity is connected.",
        "Sum": "Summarization: Based on the entities and linking relationships from the previous steps, summarize the information and organize it into a graph structured format that conforms to the Schema. Generate a draft graph as an intermediate result, formatted as a JSON list of triples."
    },
   #  "Tree": {
   #      "EE":,
   #      "EL":,
   #      "Sum":
   #  },
}

# 改成(schema, attribute)的形式
# {
#    'schema': ['Company', 'Year', 'Revenue'],\n"
#    'attribute': {'Company': 'company_name', 'Year': 'year', 'Revenue': 'amount'}
# }
PROMPTS['SCHEMA_CONSTRUCTION'] = """
You are a schema generation assistant. Given a query, analyze its task and generate a structured schema to guide entity extraction from documents. The schema should include all relevant entities and their types, even for simple queries.

### Key Rules:
1. **Entity Extraction**:
   - If the query involves a single entity (e.g., "Find the company with the highest revenue"), extract the entity type (e.g., "Company", "Revenue").
   - If the query contains composite entities (e.g., mathematical formulas or nested metrics), split them into separate entities (e.g., "Accounts Payable", "COGS", "Inventory").
2. **Language Support**:
   - The schema should support both Chinese and English queries. Use the language of the query for entity names.
3. **Output Format**:
   - Ignore any output requirements in the query. The output must strictly follow this format: (EntityType1, EntityType2, ...).
   - Adhere strictly to the format: (EntityType1, EntityType2, ...).
   - The schema should include only entity types, listed in order and separated by commas.
   - Entity types should be clear and specific (e.g., "Company", "Revenue", "Year").

### Examples:
{examples}

### Real Data:
Query: {query}

### Output:
"""


PROMPTS['SCHEMA_CONSTRUCTION_EXAMPLES'] = ["""
Example 1:
                                  
Query:  How many companies have a 'Net cash flow from financing activities change compared to the previous reporting period'  exceeding $100,000?

######################
Reason: The question asks which companies had a 'Net cash flow from financing activities change' over $100,000 compared to the previous period. We need to extract the company, 'Net cash flow from financing activities' value, and the period for comparison.
######################
Output: (Company, Net cash flow from financing activities, Period)
""",
"""
Example 2:

Query: How many companies have 'Cash Flows from Financing Activities' exceeding $1,000,000?

######################
Reason: The question asks which companies had 'Cash flow from Financing Activities' exceeding $1,000,000 in a given finance report. It requires extracting the company, 'Cash Flows from Financing Activities' values to make a comparison.
######################
Output: (Company, Cash Flows from Financing Activities)
""",
"""
Example 3:

Query: 请对上述公司财报按照 “利润总额” 进行划分，划分成：高利润(1,000,000,000.00以上)，中利润 (100,000,000.00以上且1,000,000,000.00以下)，低利润(0以上且100,000,000.00以下)，负利润(0及0以下)。同类别的划分到一个集合，不同类别的划分到不同集合。


######################
Reason: 该问题要求根据公司财报中的“利润总额”将公司进行聚类划分。需要提取各公司的“利润总额”数据进行聚类。
######################
Output: (公司, 利润总额)
""",
"""
Example 4:

Query: What is the change in Net Profit from the 2023 Annual Report to the 2024 Annual Report?

######################
Reason: The question asks for the change in 'Net Profit' between the 2023 and 2024 Annual Reports. It requires extracting the 'Net Profit' values for both years to calculate the change.
######################
Output: (Net Profit, Year)
""",
"""
Example 5:

Query: What is Amazon's FY2017 days payable outstanding (DPO)? DPO is defined as: 365 * (average accounts payable between FY2016 and FY2017) / (FY2017 COGS + change in inventory between FY2016 and FY2017). Round your answer to two decimal places. Address the question by using the line items and information shown within the balance sheet and the P&L statement.

######################
Reason: The question asks for the calculation of DPO based on multiple entities including accounts payable, COGS, inventory, and the years involved. We need to extract these entities separately for proper calculation.
######################
Output: (Company, Accounts Payable, COGS, Inventory, Year)
""",
]


PROMPTS['SCHEMA_CONSTRUCTION2'] = """
You are a schema generation assistant. Given a query, analyze its task and generate a structured schema to guide entity extraction from documents. The schema should include all relevant entities types and their attributes, even for simple queries.

### Key Rules:
1. **Entity Extraction**:
   - If the query involves a single entity (e.g., "Find the company with the highest revenue"), extract the entity type (e.g., "Company", "Revenue").
   - If the query contains composite entities (e.g., mathematical formulas or nested metrics), split them into separate entities (e.g., "Accounts Payable", "COGS", "Inventory").
2. **Language Support**:
   - The schema should support both Chinese and English queries. Use the language of the query for entity names.
3. **Output Format**:
   - Generate a JSON object with two keys: `schema` and `attribute`.
   - `schema`: List of entity types (e.g., ["Company", "Year", "Revenue"]).
   - `attribute`: Mapping from entity types to attribute names (e.g., {{"Company": "company_name", "Year": "year"}}).
   - Adhere strictly to the JSON format.
   - Attribute names should be concise English identifiers (e.g., "amount", "year").

### Examples:
{examples}

### Real Data:
Query: {query}

### Output:
"""

PROMPTS['SCHEMA_CONSTRUCTION_EXAMPLES2'] = [
    {
        "query": "What was the trend in revenue from 2018 to 2020 as per <Company A>'s annual financial reports?",
        "output": {
            "schema": ["Company", "Year", "Revenue", "Trend"],
            "attribute": {
                "Company": "company_name",
                "Year": "year",
                "Revenue": "amount"
            }
        }
    },
    {
        "query": "How many companies have a 'Net cash flow from financing activities change compared to the previous reporting period' exceeding $100,000?",
        "output": {
            "schema": ["Company", "Year", "Revenue", "Trend"],
            "attribute": {
                "Company": "company_name",
                "Year": "year",
                "Revenue": "amount"
            }
        }
    },
    {
        "query": "What is Amazon's FY2017 days payable outstanding (DPO)? DPO is defined as: 365 * (average accounts payable between FY2016 and FY2017) / (FY2017 COGS + change in inventory between FY2016 and FY2017). Round your answer to two decimal places. Address the question by using the line items and information shown within the balance sheet and the P&L statement.",
        "output": {
            "schema": ["Company", "Accounts Payable", "COGS", "Inventory", "Year"],
            "attribute": {
                "Company": "company_name",
                "Accounts Payable": "amount",
                "COGS": "amount",
                "Inventory": "amount",
                "Year": "year"
            }
        }
    },
    {
        "query": "请对上述公司财报按照 “利润总额” 进行划分，划分成：高利润(1,000,000,000.00以上)，中利润 (100,000,000.00以上且1,000,000,000.00以下)，低利润(0以上且100,000,000.00以下)，负利润(0及0以下)。同类别的划分到一个集合，不同类别的划分到不同集合。",
        "output": {
            "schema": ["公司", "利润总额"],
            "attribute": {
                "公司": "公司名字",
                "利润总额": "金额"
            }
        }
    }
]


PROMPTS["GRAPH_CONSTRUCTION"] = """
# Task Objective
Construct a graph from raw text content that strictly adheres to the provided schema. The graph should capture entities and their relationships as they appear in the text.

# Input Schema
Schema Type: {schema}

# Step-by-Step Instructions
Step 1: **Entity Extraction**:
   - Identify all entities from the raw text that match the Schema definition.
   - For each identified entity, extract the following:
       - entity_name: Name of the entity (use the same language as the input; if English, capitalize the name).
       - entity_type: One of the following types: {schema}
   - Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>).

Step 2: **Entity Linking**:
   - From the entities extracted in step 1, identify all triples of (head, relation, tail) that are *clearly related* to each other, with each triple representing a relationship.
   - For each related triple, extract the following:
       - head: Name of the source entity, as identified in Step 1.
       - relation: The type of connection or a descriptive label indicating how the head and tail are linked.
       - tail: Name of the target entity, as identified in Step 1.
   - Each triple should contain at least one, but preferably two, of the named entities identified in Step 1.
   - Format each relationship as ("relationship"{tuple_delimiter}<head>{tuple_delimiter}<relation>{tuple_delimiter}<tail>).

Step 3: **Summarization**:
   - Summarize the extracted entities and their relationships.
   - Organize the summarized information into a coherent graph structure that conforms to the Schema.
   - Generate a draft graph as an intermediate result, formatted as a JSON list of triples.

Step 4: **Data Validation**:
   - Verify that the data retains its original formats, including any special characters or units.
   - Clearly resolve pronouns to their specific names to maintain clarity.
   - Ensure that no additional or fabricated data is introduced; any missing data should be left empty.
   - If any updates are required during validation, generate the updated graph as an intermediate result.

Step 5: **Final Output Formatting**:  
   - Return the final graph representation as a single list of all entities and relationships, using **{record_delimiter}** as the list delimiter.
   - Append {completion_delimiter} at the end of the output.

# Critical Rules
- Strictly follow the Schema definitions.
- NEVER invent data not present in the raw text.
- Use "{tuple_delimiter}" as the column separator.
- Use "{record_delimiter}" as the row separator.
- Terminate with "{completion_delimiter}".
- **DO NOT perform any additional computation or reasoning; simply perform extraction.**
- **The chain-of-thought reasoning for each step should be clearly documented as intermediate results before generating the final output.**

# Examples
{examples}

# Current Task
Schema: {schema}
Raw Content: {content}

# Output Structure:\
"""

PROMPTS["TREE_CONSTRUCTION"] = """
# Task Objective
Construct a hierarchical tree from raw text content that strictly adheres to the provided schema. The tree should capture parent-child relationships between entities as they appear in the text.

# Input Schema
Schema Type: {schema}

# Step-by-Step Instructions
Step 1: **Entity Extraction**:
   - Identify all entities from the raw text that match the Schema definition.
   - For each identified entity, extract the following:
       - entity_name: Name of the entity (use the same language as the input; if English, capitalize the name).
       - entity_type: One of the following types: {schema}
   - Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>).

Step 2: **Hierarchical Relationship Extraction**:
   - From the entities extracted in Step 1, identify all hierarchical triples of (subject, predicate, object) that represent parent-child relationships.
   - For each hierarchical relationship, extract the following:
       - subject: Name of the parent entity, as identified in Step 1.
       - predicate: A hierarchical connection type (e.g., "contains", "is_parent_of", "belongs_to").
       - object: Name of the child entity, as identified in Step 1.
   - Each triple must follow a strict tree structure (no cycles).
   - Each triple should contain at least one, but preferably two, of the named entities identified in Step 1.
   - Format each relationship as ("relationship"{tuple_delimiter}<subject>{tuple_delimiter}<predicate>{tuple_delimiter}<object>).

Step 3: **Tree Summarization**:
   - Summarize the extracted entities and their hierarchical relationships.
   - Organize the summarized information into a tree structure that conforms to the Schema.
   - Generate a draft tree as an intermediate result, formatted as a JSON list of triples.

Step 4: **Data Validation**:
   - Verify the tree contains no cyclic relationships.
   - Ensure each child entity has only one parent (unless schema explicitly allows multiple inheritance).
   - Clearly resolve ambiguous references to maintain structural integrity.
   - If any validation failures occur, generate empty list as intermediate result.

Step 5: **Final Output Formatting**:
   - Return the final tree representation as a single list of all entities and hierarchical relationships, using **{record_delimiter}** as the list delimiter.
   - If no valid hierarchical relationships exist, return empty list.
   - Append {completion_delimiter} at the end of the output.

# Critical Rules
- Strictly follow the hierarchical Schema definitions.
- Relationships must form valid tree structure (root → branch → leaf).
- Use "{tuple_delimiter}" as the column separator.
- Use "{record_delimiter}" as the row separator.
- Terminate with "{completion_delimiter}".
- **Return empty list if no hierarchical relationships exist.**
- **DO NOT create relationships beyond parent-child patterns.**
- **DO NOT perform any additional computation or reasoning; simply perform extraction.**
- **The chain-of-thought reasoning for each step should be clearly documented as intermediate results before generating the final output.**

# Examples
{examples}

# Current Task
Schema: {schema}
Raw Content: {content}

# Output Structure:\
"""


PROMPTS["TABLE_CONSTRUCTION"] = """\
# Task Objective
Construct a structured table from raw text content that strictly adheres to the provided schema. The output must:
- Preserve the original language of the data
- Include table title and source metadata
- Match column definitions precisely to the schema
- Handle missing data appropriately

# Input Schema
Schema Columns: {schema}

# Step-by-Step Instructions
Step 1: **Entity Extraction**:
   - Identify relevant entities from raw content based on the schema
   - Ensure extracted entities align with schema column definitions
   - Generate an intermediate result listing all extracted entities.

Step 2: **Entity Linking**:
   For each row in raw content:
   - Map values to schema columns using exact keyword matching
   - Directly link the raw entities as they appear, without performing any additional inference.
   - If no direct match, Use contextually related values marked with [INFERRED].
   - Generate an intermediate result that outputs the relationships between entities (in the form of tuples), explaining how each entity is connected.

Step 3: **Summarization**:
   - Based on the entities and linking relationships from the previous steps, summarize the information and organize it into a table structured format that conforms to the Schema.
   - Generate a draft table as an intermediate result, ensuring that each Schema column is correctly populated.

Step 4: **Data Validation**:
   - Check that the data retains its original formats, including date/time formats, special characters, and units.
   - Strictly follow the raw content; leave missing data as empty and do not fabricate or modify any data.
   - If any updates are required during validation, generate the updated table as an intermediate result.

Step 5: **Final Output Formatting**:
   Follow this exact structure:
   ("table"{tuple_delimiter}<Title>{tuple_delimiter}<Source>{tuple_delimiter}<Description>){record_delimiter}
   ("header"{tuple_delimiter}<COLUMN_1>{tuple_delimiter}<COLUMN_2>...){record_delimiter}
   ("row"{tuple_delimiter}<VALUE_1>{tuple_delimiter}<VALUE_2>...){record_delimiter}
   ...
   {completion_delimiter}

# Critical Rules
- STRICTLY FOLLOW SCHEMA COLUMN ORDER
- NEVER invent data not present in raw content
- Use "{tuple_delimiter}" as column separator
- Use "{record_delimiter}" as row separator
- Terminate with "{completion_delimiter}"
- **DO NOT perform any additional computation or reasoning; simply perform extraction.**
- **The chain-of-thought reasoning for each step should be clearly documented as intermediate results before generating the final table.**


# Current Task
Schema: {schema}
Raw Content: {content}

# Output Structure:\
""" 

PROMPTS["TABLE_CONSTRUCTION2"] = """\
# Task Objective
Construct a structured table from raw text content that strictly adheres to the provided schema and its corresponding attribute patterns. The output must:
- Preserve the original language of the data
- Include table title and source metadata
- Precisely follow the column order defined in the schema and use the associated attribute identifiers for each column
- Handle missing data appropriately

# Input Schema and Attributes Patterns
Schema: {schema}
Attributes Patterns: {attribute}

# Step-by-Step Instructions
Step 1: **Entity Extraction**:
   - Identify relevant entities from the raw content based on the provided schema.
   - Ensure that the extracted entities align with both the schema columns and the corresponding attribute patterns.
   - Generate an intermediate result listing all extracted entities.

Step 2: **Entity Linking**:
   For each row in the raw content:
   - Map the extracted values to the schema columns using exact keyword matching.
   - **DO NOT perform any additional computation or reasoning; simply link the raw entities as they appear.**
   - If no direct match exists, use contextually related values marked with [INFERRED].
   - Generate an intermediate result that shows the relationships between entities (in the form of tuples), explaining how each entity is connected and mapped to its attribute.

Step 3: **Summarization**:
   - Based on the extracted entities and their linking relationships, summarize the information into a draft table that conforms to the schema and attribute pattern.
   - Generate a draft table as an intermediate result, ensure that each column (as defined by the schema) is correctly populated using its associated attribute identifier.

Step 4: **Data Validation**:
   - Check that the data retains its original formats, including date/time formats, special characters, and units.
   - Strictly follow the raw content; leave missing data as empty without fabricating or modifying any data.
   - If any updates are required during validation, generate an updated table as an intermediate result.

Step 5: **Final Output Formatting**:
   Follow this exact structure:
   ("table"{tuple_delimiter}<Title>{tuple_delimiter}<Description>){record_delimiter}
   ("header"{tuple_delimiter}<COLUMN_1>{tuple_delimiter}<COLUMN_2>...){record_delimiter}
   ("row"{tuple_delimiter}<VALUE_1>{tuple_delimiter}<VALUE_2>...){record_delimiter}
   ...
   {completion_delimiter}

# Critical Rules
- STRICTLY FOLLOW THE SCHEMA COLUMN ORDER
- EXTRACTED VALUES MUST MATCH ATTRIBUTE PATTERNS:
- NEVER invent data not present in the raw content.
- Use "{tuple_delimiter}" as the column separator.
- Use "{record_delimiter}" as the row separator.
- Terminate the output with "{completion_delimiter}".
- **DO NOT perform any additional computation or reasoning; simply perform extraction based on the provided schema and attribute mapping.**
- **Document the chain-of-thought reasoning for each step as intermediate results before generating the final table.**

# Examples
{examples}

# Current Task
Query: {query}
Raw Content: {content}

# Output Structure:\
"""

PROMPTS["TABLE_CONSTRUCTION3"] = """\
# Task Objective
Construct a structured table from raw text content that strictly adheres to the provided schema and its corresponding attribute patterns. The output must:
- Preserve the original language of the data
- Include table title and source metadata
- Precisely follow the column order defined in the schema and use the associated attribute identifiers for each column
- Handle missing data appropriately

# Critical Rules
1. **No Reasoning Allowed**:
   - If a schema value requires logical inference/computation → Leave blank
   - If a value is not explicitly stated → Do NOT guess or infer
   - Only use directly stated values matching attribute patterns

2. **Schema Adherence**:
   - Column order: {schema}
   - Value Patterns: {attribute}

3. **Generate Intermediate Rresults**
   - Document the chain-of-thought reasoning for each step as intermediate results before generating the final table.

# Step-by-Step Instructions
Step 1: **Entity Extraction**:
   - Identify relevant entities from the raw content based on the provided schema.
   - Ensure that the extracted entities align with both the schema columns and the corresponding attribute patterns.
   - Generate an intermediate result listing all extracted entities.
   - Output:
     ```
     **Step 1: Entity Extraction**
     [Company] (company_name): Microsoft
     [Year] (year): 2023
     [Revenue] (amount): $80B
     ```

Step 2: **Entity Linking**:
   For each row in the raw content:
   - Map the extracted values to the schema columns using exact keyword matching.
   - If no direct match exists, use contextually related values marked with [INFERRED].
   - Generate an intermediate result that shows the relationships between entities (in the form of tuples), explaining how each entity is connected and mapped to its attribute.
   - OUTPUT:
     ```
     **Step 2: Entity Linking**
     ("Microsoft", "2023", "$80B") → (Company, Year, Revenue)
     ```

Step 3: **Summarization**:
   - Based on the extracted entities and their linking relationships, summarize the information into a draft table that conforms to the schema and attribute pattern.
   - Generate a draft table as an intermediate result, ensure that each column (as defined by the schema) is correctly populated using its associated attribute identifier.
   - OUTPUT:
      ```
      **Step 3: Summarization**
      |  Company  | Year | Revenue |
      |-----------|------|---------|
      | Microsoft | 2023 |   $80B  |
      ```

Step 4: **Data Validation**:
   - Check that the data retains its original formats, including date/time formats, special characters, and units.
   - Strictly follow the raw content; leave missing data as empty without fabricating or modifying any data.
   - If any updates are required during validation, generate an updated table as an intermediate result.

Step 5: **Final Output Formatting**:
   Follow this exact structure:
   ("table"{tuple_delimiter}<Title>{tuple_delimiter}<Description>){record_delimiter}
   ("header"{tuple_delimiter}<COLUMN_1>{tuple_delimiter}<COLUMN_2>...){record_delimiter}
   ("row"{tuple_delimiter}<VALUE_1>{tuple_delimiter}<VALUE_2>...){record_delimiter}
   ...
   {completion_delimiter}

# Validation Protocol
- STRICTLY FOLLOW THE SCHEMA COLUMN ORDER
- EXTRACTED VALUES MUST MATCH ATTRIBUTE PATTERNS:
- NEVER invent data not present in the raw content.

# Current Task
Schema: {schema}
Attribute Patterns: {attribute}
Content: {content}

# Output Structure:\
"""

PROMPTS["TABLE_CONSTRUCTION4"] = """\
# Task Objective
Construct a structured table from raw text content that strictly adheres to the provided schema. The output must:
- Preserve the original language of the data
- Include table title and source metadata
- Match column definitions precisely to the schema
- Handle missing data appropriately

# Critical Rules
1. **No Reasoning Allowed**:
   - If a schema value requires logical inference/computation → Leave blank
   - If a value is not explicitly stated → Do NOT guess or infer
   - Only use directly stated values matching attribute patterns

2. **Input Schema**:
   - Schema Columns: {schema}

3. **Generate Intermediate Rresults**
   - Document the chain-of-thought reasoning for each step as intermediate results before generating the final table.

# Step-by-Step Instructions
Step 1: **Entity Extraction**:
   - Identify relevant entities from the raw content based on the provided schema.
   - Ensure that the extracted entities align with both the schema columns.
   - Generate an intermediate result listing all extracted entities.
   - Output:
     ```
     **Step 1: Entity Extraction**
     [Company]: Microsoft
     [Year]: 2023
     [Revenue]: $80B
     ```

Step 2: **Entity Linking**:
   For each row in the raw content:
   - Map the extracted values to the schema columns using exact keyword matching.
   - If no direct match exists, use contextually related values marked with [INFERRED].
   - Generate an intermediate result that shows the relationships between entities (in the form of tuples), explaining how each entity is connected.
   - OUTPUT:
     ```
     **Step 2: Entity Linking**
     ("Microsoft", "2023", "$80B") → (Company, Year, Revenue)
     ```

Step 3: **Summarization**:
   - Based on the entities and linking relationships from the previous steps, summarize the information and organize it into a table structured format that conforms to the Schema.
   - Generate a draft table as an intermediate result, ensuring that each Schema column is correctly populated.
   - OUTPUT:
      ```
      **Step 3: Summarization**
      |  Company  | Year | Revenue |
      |-----------|------|---------|
      | Microsoft | 2023 |   $80B  |
      ```

Step 4: **Final Output Formatting**:
   Follow this exact structure:
   ("table"{tuple_delimiter}<Title>{tuple_delimiter}<Description>){record_delimiter}
   ("header"{tuple_delimiter}<COLUMN_1>{tuple_delimiter}<COLUMN_2>...){record_delimiter}
   ("row"{tuple_delimiter}<VALUE_1>{tuple_delimiter}<VALUE_2>...){record_delimiter}
   ...
   {completion_delimiter}

# Validation Protocol
- STRICTLY FOLLOW THE SCHEMA COLUMN ORDER
- EXTRACTED VALUES MUST MATCH ATTRIBUTE PATTERNS:
- NEVER invent data not present in the raw content.

# Current Task
Schema: {schema}
Content: {content}

# Output Structure:\
"""

PROMPTS["TREE_CONSTRUCTION_EXAMPLES"] = [
# """
# Example 1:
                                  
# Schema: [Company, Net cash flow from financing activities, Period]
# Text:
# DIVALL INSURED INCOME PROPERTIES 2 LIMITED PARTNERSHIP\n\n                     
# CASH FLOWS USED IN FINANCING ACTIVITIES:                                                                                           \n\nCash distributions to Limited Partners                                                           (234,495)                (475,001)\n\nNet cash used in financing activities                                                            (234,495)\n\n                (475,001)\n\n       
# For the three month period ended March 31, 2024 and 2023 cash flows used in financing activities were $234,497 and $475,001, respectively, and consisted of aggregate general and limited partner distributions. Distributions have been and are expected to continue to be made in accordance with the Partnership Agreement.
                                                                       
# BIOLARGO, INC. AND SUBSIDIARIES\n\nCONDENSED CONSOLIDATED STATEMENTS OF CASH FLOWS\n\n(in thousands, except for share and per share data)\n\n(unaudited)\n\n \n\n                                                                                                     Three Months Ended March 31,      \n\n                                                                                                              2024               2023 \n\n  Cash flows from operating activities                                                                                               \n\n  Net loss                                                                                       $           (775) $            (494)\n\n  Adjustments to reconcile net loss to net cash provided by (used in) operating activities:                                          \n\n  Stock option compensation expense                                                                           488                256 \n\n  Common stock issued for services                                                                            135                207 \n\n  Amortization of right-of-use operating lease assets                                                          24                  — \n\n  Interest expense related to amortization of the discount on note payable                                      —                  3 \n\n  Fair value of warrant issued for interest                                                                     —                 30 \n\n  Loss on investment in South Korean joint venture                                                              1                  6 \n\n  Depreciation expense                                                                                         36                 22 \n\n  Changes in assets and liabilities:                                                                                                 \n\n  Accounts receivable                                                                                         139               (316)\n\n  Inventories                                                                                                 (98)               (17)\n\n  Prepaid expenses and other assets                                                                           (58)                25 \n\n  Accounts payable and accrued expenses                                                                       252                284 \n\n  Deposits                                                                                                    109                (71)\n\n  Clyra accounts payable and accrued expenses                                                                 289                 14 \n\n  Contract liabilities                                                                                        (42)                 4 \n\n  Lease liability, net                                                                                        (19)                 4 \n\n  Net cash provided by (used in) operating activities                                                         481                (43)\n\n  Cash flows from investing activities                                                                                               \n\n  Equipment purchases                                                                                        (863)               (48)\n\n  Net cash used in investing activities                                                                      (863)               (48)\n\n  Cash flows from financing activities                                                                                               \n\n  Proceeds from sale of common stock, net of commissions                                                      488                800 \n\n  Proceeds from warrant exercise                                                                               75                  — \n\n  Proceeds from sale of BETI common stock                                                                      50                550 \n\n  Repayment of debt obligations                                                                                (5)               (50)\n\n  Repayment by Clyra debt obligations                                                                           —                (15)\n\n  Proceeds from sale of Clyra Medical preferred stock                                                           —                225 \n\n  Proceeds from sale of Clyra Medical common stock                                                            475                  — \n\n  Net cash provided by financing activities                                                                 1,083              1,510 \n\n  Net effect of foreign currency translation                                                                   96                 (6)\n\n  Net change in cash                                                                                          797              1,413 \n\n  Cash and cash equivalents at beginning of period                                                          3,539              1,851 \n\n  Cash and cash equivalents at end of period                                                     $          4,336  $           3,264 \n\n  Supplemental disclosures of cash flow information                                                                                  \n\n  Cash paid during the period for:                                                                                                   \n\n  Interest                                                                                       $             12  $              15 \n\n  Income taxes                                                                                   $              —  $               5 \n\n  Short-term lease payments not included in lease liability                                      $             12  $              13 \n\n  Non-cash investing and financing activities                                                                                        \n\n  Equipment added using capital lease                                                            $              —  $              80 \n\n  Conversion of Clyra common stock to BioLargo common stock                                      $              —  $             100 \n\n  Allocation of noncontrolling interest                                                          $            399  $             467 \n\n  Preferred Series A Dividend                                                                    $             86  $               — \n\n \n\nThe accompanying notes are an integral part of these unaudited condensed consolidated financial statements.

# ######################
# Output:
# ("entity"{tuple_delimiter}"DIVALL INSURED INCOME PROPERTIES 2 LIMITED PARTNERSHIP"{tuple_delimiter}"Company"){record_delimiter}                                         
# ("entity"{tuple_delimiter}"$234,497"{tuple_delimiter}"Net cash flow from financing activities"){record_delimiter}                                         
# ("entity"{tuple_delimiter}"$475,001"{tuple_delimiter}"Net cash flow from financing activities"){record_delimiter}                                         
# ("entity"{tuple_delimiter}"March 31, 2024"{tuple_delimiter}"Period"){record_delimiter}                                         
# ("entity"{tuple_delimiter}"March 31, 2023"{tuple_delimiter}"Period"){record_delimiter}
# ("entity"{tuple_delimiter}"BIOLARGO, INC. AND SUBSIDIARIES"{tuple_delimiter}"Company"){record_delimiter}                                         
# ("entity"{tuple_delimiter}"1,083"{tuple_delimiter}"Net cash flow from financing activities"){record_delimiter}                                         
# ("entity"{tuple_delimiter}"1,510"{tuple_delimiter}"Net cash flow from financing activities"){record_delimiter}                                                                                                                         
# ("relationship"{tuple_delimiter}"DIVALL INSURED INCOME PROPERTIES 2 LIMITED PARTNERSHIP"{tuple_delimiter}"has net cash flow from financial activities"{tuple_delimiter}"$234,497"){record_delimiter}
# ("relationship"{tuple_delimiter}"DIVALL INSURED INCOME PROPERTIES 2 LIMITED PARTNERSHIP"{tuple_delimiter}"has net cash flow from financial activities"{tuple_delimiter}"$475,001"){record_delimiter}
# ("relationship"{tuple_delimiter}"$234,497"{tuple_delimiter}"reported on"{tuple_delimiter}"March 31, 2024"){record_delimiter}
# ("relationship"{tuple_delimiter}"$475,001"{tuple_delimiter}"reported on"{tuple_delimiter}"March 31, 2023"){record_delimiter}
# ("relationship"{tuple_delimiter}"BIOLARGO, INC. AND SUBSIDIARIES"{tuple_delimiter}"has net cash flow from financial activities"{tuple_delimiter}"1,083"){record_delimiter}
# ("relationship"{tuple_delimiter}"BIOLARGO, INC. AND SUBSIDIARIES"{tuple_delimiter}"has net cash flow from financial activities"{tuple_delimiter}"1,510"){record_delimiter}
# ("relationship"{tuple_delimiter}"1,083"{tuple_delimiter}"reported on"{tuple_delimiter}"March 31, 2024"){record_delimiter}
# ("relationship"{tuple_delimiter}"1,510"{tuple_delimiter}"reported on"{tuple_delimiter}"March 31, 2023"){completion_delimiter}
# ######################          
# """,
]



PROMPTS["TABLE_CONSTRUCTION_EXAMPLES"] = [
"""
Example 1:
Query: How many companies have a 'Net cash flow from financing activities change compared to the previous reporting period'  exceeding $100,000?                
Schema: [Company, Net cash flow from financing activities, Period]
Raw Content:
DIVALL INSURED INCOME PROPERTIES 2 LIMITED PARTNERSHIP\n\n                     
CASH FLOWS USED IN FINANCING ACTIVITIES:                                                                                           \n\nCash distributions to Limited Partners                                                           (234,495)                (475,001)\n\nNet cash used in financing activities                                                            (234,495)\n\n                (475,001)\n\n       
For the three month period ended March 31, 2024 and 2023 cash flows used in financing activities were $234,497 and $475,001, respectively, and consisted of aggregate general and limited partner distributions. Distributions have been and are expected to continue to be made in accordance with the Partnership Agreement.
                                                                       
BIOLARGO, INC. AND SUBSIDIARIES\n\nCONDENSED CONSOLIDATED STATEMENTS OF CASH FLOWS\n\n(in thousands, except for share and per share data)\n\n(unaudited)\n\n \n\n                                                                                                     Three Months Ended March 31,      \n\n                                                                                                              2024               2023 \n\n  Cash flows from operating activities                                                                                               \n\n  Net loss                                                                                       $           (775) $            (494)\n\n  Adjustments to reconcile net loss to net cash provided by (used in) operating activities:                                          \n\n  Stock option compensation expense                                                                           488                256 \n\n  Common stock issued for services                                                                            135                207 \n\n  Amortization of right-of-use operating lease assets                                                          24                  — \n\n  Interest expense related to amortization of the discount on note payable                                      —                  3 \n\n  Fair value of warrant issued for interest                                                                     —                 30 \n\n  Loss on investment in South Korean joint venture                                                              1                  6 \n\n  Depreciation expense                                                                                         36                 22 \n\n  Changes in assets and liabilities:                                                                                                 \n\n  Accounts receivable                                                                                         139               (316)\n\n  Inventories                                                                                                 (98)               (17)\n\n  Prepaid expenses and other assets                                                                           (58)                25 \n\n  Accounts payable and accrued expenses                                                                       252                284 \n\n  Deposits                                                                                                    109                (71)\n\n  Clyra accounts payable and accrued expenses                                                                 289                 14 \n\n  Contract liabilities                                                                                        (42)                 4 \n\n  Lease liability, net                                                                                        (19)                 4 \n\n  Net cash provided by (used in) operating activities                                                         481                (43)\n\n  Cash flows from investing activities                                                                                               \n\n  Equipment purchases                                                                                        (863)               (48)\n\n  Net cash used in investing activities                                                                      (863)               (48)\n\n  Cash flows from financing activities                                                                                               \n\n  Proceeds from sale of common stock, net of commissions                                                      488                800 \n\n  Proceeds from warrant exercise                                                                               75                  — \n\n  Proceeds from sale of BETI common stock                                                                      50                550 \n\n  Repayment of debt obligations                                                                                (5)               (50)\n\n  Repayment by Clyra debt obligations                                                                           —                (15)\n\n  Proceeds from sale of Clyra Medical preferred stock                                                           —                225 \n\n  Proceeds from sale of Clyra Medical common stock                                                            475                  — \n\n  Net cash provided by financing activities                                                                 1,083              1,510 \n\n  Net effect of foreign currency translation                                                                   96                 (6)\n\n  Net change in cash                                                                                          797              1,413 \n\n  Cash and cash equivalents at beginning of period                                                          3,539              1,851 \n\n  Cash and cash equivalents at end of period                                                     $          4,336  $           3,264 \n\n  Supplemental disclosures of cash flow information                                                                                  \n\n  Cash paid during the period for:                                                                                                   \n\n  Interest                                                                                       $             12  $              15 \n\n  Income taxes                                                                                   $              —  $               5 \n\n  Short-term lease payments not included in lease liability                                      $             12  $              13 \n\n  Non-cash investing and financing activities                                                                                        \n\n  Equipment added using capital lease                                                            $              —  $              80 \n\n  Conversion of Clyra common stock to BioLargo common stock                                      $              —  $             100 \n\n  Allocation of noncontrolling interest                                                          $            399  $             467 \n\n  Preferred Series A Dividend                                                                    $             86  $               — \n\n \n\nThe accompanying notes are an integral part of these unaudited condensed consolidated financial statements.

######################
Output:
("table"{tuple_delimiter}"Net Cash Flow Comparison of Major Companies"{tuple_delimiter}"This table presents the net cash flow from financing activities for DIVALL Insured Income Properties 2 Limited Partnership and BioLargo for the periods ending March 31, 2024, and March 31, 2023. Data 
includes changes in net cash flows above $100,000."){record_delimiter}
("header"{tuple_delimiter}"Company"{tuple_delimiter}"Net cash flow from financing activities"{tuple_delimiter}"Period"){record_delimiter}
("row"{tuple_delimiter}"DIVALL INSURED INCOME PROPERTIES 2 LIMITED PARTNERSHIP"{tuple_delimiter}"$234,497"{tuple_delimiter}"March 31, 2024"){record_delimiter}
("row"{tuple_delimiter}"DIVALL INSURED INCOME PROPERTIES 2 LIMITED PARTNERSHIP"{tuple_delimiter}"$475,001"{tuple_delimiter}"March 31, 2023"){record_delimiter}
("row"{tuple_delimiter}"BIOLARGO, INC. AND SUBSIDIARIES"{tuple_delimiter}"1,083"{tuple_delimiter}"March 31, 2024"){record_delimiter}
("row"{tuple_delimiter}"BIOLARGO, INC. AND SUBSIDIARIES"{tuple_delimiter}"1,510"{tuple_delimiter}"March 31, 2023"){completion_delimiter}
######################
""",
]

PROMPTS["TABLE_CONSTRUCTION_EXAMPLES2"] = [
"""
Example 1:
Query: How many companies have a 'Net cash flow from financing activities change compared to the previous reporting period'  exceeding $100,000?                
Schema: [Company, Net cash flow from financing activities, Period]
Attributes Patterns: {'Company': 'company_name', 'Net cash flow from financing activities': 'amount', 'Period': 'time'}

Raw Content:
DIVALL INSURED INCOME PROPERTIES 2 LIMITED PARTNERSHIP\n\n                     
CASH FLOWS USED IN FINANCING ACTIVITIES:                                                                                           \n\nCash distributions to Limited Partners                                                           (234,495)                (475,001)\n\nNet cash used in financing activities                                                            (234,495)\n\n                (475,001)\n\n       
For the three month period ended March 31, 2024 and 2023 cash flows used in financing activities were $234,497 and $475,001, respectively, and consisted of aggregate general and limited partner distributions. Distributions have been and are expected to continue to be made in accordance with the Partnership Agreement.
                                                                       
BIOLARGO, INC. AND SUBSIDIARIES\n\nCONDENSED CONSOLIDATED STATEMENTS OF CASH FLOWS\n\n(in thousands, except for share and per share data)\n\n(unaudited)\n\n \n\n                                                                                                     Three Months Ended March 31,      \n\n                                                                                                              2024               2023 \n\n  Cash flows from operating activities                                                                                               \n\n  Net loss                                                                                       $           (775) $            (494)\n\n  Adjustments to reconcile net loss to net cash provided by (used in) operating activities:                                          \n\n  Stock option compensation expense                                                                           488                256 \n\n  Common stock issued for services                                                                            135                207 \n\n  Amortization of right-of-use operating lease assets                                                          24                  — \n\n  Interest expense related to amortization of the discount on note payable                                      —                  3 \n\n  Fair value of warrant issued for interest                                                                     —                 30 \n\n  Loss on investment in South Korean joint venture                                                              1                  6 \n\n  Depreciation expense                                                                                         36                 22 \n\n  Changes in assets and liabilities:                                                                                                 \n\n  Accounts receivable                                                                                         139               (316)\n\n  Inventories                                                                                                 (98)               (17)\n\n  Prepaid expenses and other assets                                                                           (58)                25 \n\n  Accounts payable and accrued expenses                                                                       252                284 \n\n  Deposits                                                                                                    109                (71)\n\n  Clyra accounts payable and accrued expenses                                                                 289                 14 \n\n  Contract liabilities                                                                                        (42)                 4 \n\n  Lease liability, net                                                                                        (19)                 4 \n\n  Net cash provided by (used in) operating activities                                                         481                (43)\n\n  Cash flows from investing activities                                                                                               \n\n  Equipment purchases                                                                                        (863)               (48)\n\n  Net cash used in investing activities                                                                      (863)               (48)\n\n  Cash flows from financing activities                                                                                               \n\n  Proceeds from sale of common stock, net of commissions                                                      488                800 \n\n  Proceeds from warrant exercise                                                                               75                  — \n\n  Proceeds from sale of BETI common stock                                                                      50                550 \n\n  Repayment of debt obligations                                                                                (5)               (50)\n\n  Repayment by Clyra debt obligations                                                                           —                (15)\n\n  Proceeds from sale of Clyra Medical preferred stock                                                           —                225 \n\n  Proceeds from sale of Clyra Medical common stock                                                            475                  — \n\n  Net cash provided by financing activities                                                                 1,083              1,510 \n\n  Net effect of foreign currency translation                                                                   96                 (6)\n\n  Net change in cash                                                                                          797              1,413 \n\n  Cash and cash equivalents at beginning of period                                                          3,539              1,851 \n\n  Cash and cash equivalents at end of period                                                     $          4,336  $           3,264 \n\n  Supplemental disclosures of cash flow information                                                                                  \n\n  Cash paid during the period for:                                                                                                   \n\n  Interest                                                                                       $             12  $              15 \n\n  Income taxes                                                                                   $              —  $               5 \n\n  Short-term lease payments not included in lease liability                                      $             12  $              13 \n\n  Non-cash investing and financing activities                                                                                        \n\n  Equipment added using capital lease                                                            $              —  $              80 \n\n  Conversion of Clyra common stock to BioLargo common stock                                      $              —  $             100 \n\n  Allocation of noncontrolling interest                                                          $            399  $             467 \n\n  Preferred Series A Dividend                                                                    $             86  $               — \n\n \n\nThe accompanying notes are an integral part of these unaudited condensed consolidated financial statements.

######################
Output:
("table"{tuple_delimiter}"Net Cash Flow Comparison of Major Companies"{tuple_delimiter}"This table presents the net cash flow from financing activities for DIVALL Insured Income Properties 2 Limited Partnership and BioLargo for the periods ending March 31, 2024, and March 31, 2023. Data 
includes changes in net cash flows above $100,000."){record_delimiter}
("header"{tuple_delimiter}"Company"{tuple_delimiter}"Net cash flow from financing activities"{tuple_delimiter}"Period"){record_delimiter}
("row"{tuple_delimiter}"DIVALL INSURED INCOME PROPERTIES 2 LIMITED PARTNERSHIP"{tuple_delimiter}"$234,497"{tuple_delimiter}"March 31, 2024"){record_delimiter}
("row"{tuple_delimiter}"DIVALL INSURED INCOME PROPERTIES 2 LIMITED PARTNERSHIP"{tuple_delimiter}"$475,001"{tuple_delimiter}"March 31, 2023"){record_delimiter}
("row"{tuple_delimiter}"BIOLARGO, INC. AND SUBSIDIARIES"{tuple_delimiter}"1,083"{tuple_delimiter}"March 31, 2024"){record_delimiter}
("row"{tuple_delimiter}"BIOLARGO, INC. AND SUBSIDIARIES"{tuple_delimiter}"1,510"{tuple_delimiter}"March 31, 2023"){completion_delimiter}
######################
""",
]


PROMPTS["GRAPH_CONSTRUCTION_EXAMPLES"] = ["""
Example 1:
                                  
Schema: [Company, Net cash flow from financing activities, Period]
Text:
DIVALL INSURED INCOME PROPERTIES 2 LIMITED PARTNERSHIP\n\n                     
CASH FLOWS USED IN FINANCING ACTIVITIES:                                                                                           \n\nCash distributions to Limited Partners                                                           (234,495)                (475,001)\n\nNet cash used in financing activities                                                            (234,495)\n\n                (475,001)\n\n       
For the three month period ended March 31, 2024 and 2023 cash flows used in financing activities were $234,497 and $475,001, respectively, and consisted of aggregate general and limited partner distributions. Distributions have been and are expected to continue to be made in accordance with the Partnership Agreement.
                                                                       
BIOLARGO, INC. AND SUBSIDIARIES\n\nCONDENSED CONSOLIDATED STATEMENTS OF CASH FLOWS\n\n(in thousands, except for share and per share data)\n\n(unaudited)\n\n \n\n                                                                                                     Three Months Ended March 31,      \n\n                                                                                                              2024               2023 \n\n  Cash flows from operating activities                                                                                               \n\n  Net loss                                                                                       $           (775) $            (494)\n\n  Adjustments to reconcile net loss to net cash provided by (used in) operating activities:                                          \n\n  Stock option compensation expense                                                                           488                256 \n\n  Common stock issued for services                                                                            135                207 \n\n  Amortization of right-of-use operating lease assets                                                          24                  — \n\n  Interest expense related to amortization of the discount on note payable                                      —                  3 \n\n  Fair value of warrant issued for interest                                                                     —                 30 \n\n  Loss on investment in South Korean joint venture                                                              1                  6 \n\n  Depreciation expense                                                                                         36                 22 \n\n  Changes in assets and liabilities:                                                                                                 \n\n  Accounts receivable                                                                                         139               (316)\n\n  Inventories                                                                                                 (98)               (17)\n\n  Prepaid expenses and other assets                                                                           (58)                25 \n\n  Accounts payable and accrued expenses                                                                       252                284 \n\n  Deposits                                                                                                    109                (71)\n\n  Clyra accounts payable and accrued expenses                                                                 289                 14 \n\n  Contract liabilities                                                                                        (42)                 4 \n\n  Lease liability, net                                                                                        (19)                 4 \n\n  Net cash provided by (used in) operating activities                                                         481                (43)\n\n  Cash flows from investing activities                                                                                               \n\n  Equipment purchases                                                                                        (863)               (48)\n\n  Net cash used in investing activities                                                                      (863)               (48)\n\n  Cash flows from financing activities                                                                                               \n\n  Proceeds from sale of common stock, net of commissions                                                      488                800 \n\n  Proceeds from warrant exercise                                                                               75                  — \n\n  Proceeds from sale of BETI common stock                                                                      50                550 \n\n  Repayment of debt obligations                                                                                (5)               (50)\n\n  Repayment by Clyra debt obligations                                                                           —                (15)\n\n  Proceeds from sale of Clyra Medical preferred stock                                                           —                225 \n\n  Proceeds from sale of Clyra Medical common stock                                                            475                  — \n\n  Net cash provided by financing activities                                                                 1,083              1,510 \n\n  Net effect of foreign currency translation                                                                   96                 (6)\n\n  Net change in cash                                                                                          797              1,413 \n\n  Cash and cash equivalents at beginning of period                                                          3,539              1,851 \n\n  Cash and cash equivalents at end of period                                                     $          4,336  $           3,264 \n\n  Supplemental disclosures of cash flow information                                                                                  \n\n  Cash paid during the period for:                                                                                                   \n\n  Interest                                                                                       $             12  $              15 \n\n  Income taxes                                                                                   $              —  $               5 \n\n  Short-term lease payments not included in lease liability                                      $             12  $              13 \n\n  Non-cash investing and financing activities                                                                                        \n\n  Equipment added using capital lease                                                            $              —  $              80 \n\n  Conversion of Clyra common stock to BioLargo common stock                                      $              —  $             100 \n\n  Allocation of noncontrolling interest                                                          $            399  $             467 \n\n  Preferred Series A Dividend                                                                    $             86  $               — \n\n \n\nThe accompanying notes are an integral part of these unaudited condensed consolidated financial statements.

######################
Output:
("entity"{tuple_delimiter}"DIVALL INSURED INCOME PROPERTIES 2 LIMITED PARTNERSHIP"{tuple_delimiter}"Company"){record_delimiter}                                         
("entity"{tuple_delimiter}"$234,497"{tuple_delimiter}"Net cash flow from financing activities"){record_delimiter}                                         
("entity"{tuple_delimiter}"$475,001"{tuple_delimiter}"Net cash flow from financing activities"){record_delimiter}                                         
("entity"{tuple_delimiter}"March 31, 2024"{tuple_delimiter}"Period"){record_delimiter}                                         
("entity"{tuple_delimiter}"March 31, 2023"{tuple_delimiter}"Period"){record_delimiter}
("entity"{tuple_delimiter}"BIOLARGO, INC. AND SUBSIDIARIES"{tuple_delimiter}"Company"){record_delimiter}                                         
("entity"{tuple_delimiter}"1,083"{tuple_delimiter}"Net cash flow from financing activities"){record_delimiter}                                         
("entity"{tuple_delimiter}"1,510"{tuple_delimiter}"Net cash flow from financing activities"){record_delimiter}                                                                                                                         
("relationship"{tuple_delimiter}"DIVALL INSURED INCOME PROPERTIES 2 LIMITED PARTNERSHIP"{tuple_delimiter}"has net cash flow from financial activities"{tuple_delimiter}"$234,497"){record_delimiter}
("relationship"{tuple_delimiter}"DIVALL INSURED INCOME PROPERTIES 2 LIMITED PARTNERSHIP"{tuple_delimiter}"has net cash flow from financial activities"{tuple_delimiter}"$475,001"){record_delimiter}
("relationship"{tuple_delimiter}"$234,497"{tuple_delimiter}"reported on"{tuple_delimiter}"March 31, 2024"){record_delimiter}
("relationship"{tuple_delimiter}"$475,001"{tuple_delimiter}"reported on"{tuple_delimiter}"March 31, 2023"){record_delimiter}
("relationship"{tuple_delimiter}"BIOLARGO, INC. AND SUBSIDIARIES"{tuple_delimiter}"has net cash flow from financial activities"{tuple_delimiter}"1,083"){record_delimiter}
("relationship"{tuple_delimiter}"BIOLARGO, INC. AND SUBSIDIARIES"{tuple_delimiter}"has net cash flow from financial activities"{tuple_delimiter}"1,510"){record_delimiter}
("relationship"{tuple_delimiter}"1,083"{tuple_delimiter}"reported on"{tuple_delimiter}"March 31, 2024"){record_delimiter}
("relationship"{tuple_delimiter}"1,510"{tuple_delimiter}"reported on"{tuple_delimiter}"March 31, 2023"){completion_delimiter}
######################          
""",
# """
# Example 2:

# Entity Type: [公司，利润总额]
# Text:
# 证券代码：300300                证券简称：海峡创新              公告编号： 2024-020
# |本公司及董事会全体成员保证信息披露的内容真实、准确、完整，没有虚假记载、误<br>导性陈述或重大遗漏。|
# |---|\n\n
# 2、合并利润表\n单位：元\n|项目|本期发生额|上期发生额|\n|---|---|---|\n|一、营业总收入|22,874,538.73|21,901,242.03|\n|其中：营业收入|22,874,538.73|21,901,242.03|\n|利息收入|-|-|\n|已赚保费|-|-|\n|手续费及佣金收入|-|-|\n|二、营业总成本|47,364,137.47|46,454,937.51|\n|其中：营业成本|25,899,921.91|23,900,071.63|\n|利息支出|-|-|\n|手续费及佣金支出|-|-|\n|退保金|-|-|\n|赔付支出净额|-|-|\n|提取保险责任准备金净额|-|-|\n|保单红利支出|-|-|\n|分保费用|-|-|\n|税金及附加|422,316.54|712,043.99|\n|销售费用|736,696.78|4,469,354.08|\n|管理费用|6,272,958.43|6,340,256.25|\n|研发费用|2,151,990.01|2,003,131.20|\n|财务费用|11,880,253.80|9,030,080.36|\n|其中：利息费用|11,996,238.36|7,446,328.42|\n|利息收入|19,785.65|48,726.39|\n|加：其他收益|15,870.77|186,268.45|\n|投资收益（损失以“－”号填列）|4,687,012.78|5,982,851.72|\n|其中：对联营企业和合营企业的投资收益|-|5,982,851.72|\n|以摊余成本计量的金融资产终止确认收益|-|-|\n|汇兑收益（损失以“-”号填列）|-|-|\n|净敞口套期收益（损失以“－”号填列）|-|-|\n|公允价值变动收益（损失以“－”号填列）|-26,845.01|-749,373.56|\n|信用减值损失（损失以“-”号填列）|-1,364,729.22|1,094,525.85|\n|资产减值损失（损失以“-”号填列）|-1,930,910.10|-728,161.06||资产处置收益（损失以“-”号填列）|1,102,461.85|0.00|\n|---|---|---|\n|三、营业利润（亏损以“－”号填列）|-22,006,737.67|-18,767,584.08|\n|加：营业外收入|143,171.96|288.23|\n|减：营业外支出|2,625,078.22|6,448.22|\n|四、利润总额（亏损总额以“－”号填列）|-24,488,643.93|-18,773,744.07|\n|减：所得税费用|3,948,682.34|-188,500.23|\n|五、净利润（净亏损以“－”号填列）|-28,437,326.27|-18,585,243.84|\n|（一）按经营持续性分类|-|-|\n|1.持续经营净利润（净亏损以“－”号填列）|-28,437,326.27|-18,585,243.84|\n|2.终止经营净利润（净亏损以“－”号填列）|0.00|0.00|\n|（二）按所有权归属分类|-|-|\n|1.归属于母公司所有者的净利润|-28,346,980.33|-18,568,258.84|\n|2.少数股东损益|-90,345.94|-16,985.00|\n|六、其他综合收益的税后净额|-145,529.81|396,585.55|\n|归属母公司所有者的其他综合收益的税后净额|-145,529.81|396,585.55|\n|（一）不能重分类进损益的其他综合收益|-|-|\n|1.重新计量设定受益计划变动额|-|-|\n|2.权益法下不能转损益的其他综合收益|-|-|\n|3.其他权益工具投资公允价值变动|-|-|\n|4.企业自身信用风险公允价值变动|-|-|\n|5.其他|-|-|\n|（二）将重分类进损益的其他综合收益|-145,529.81|396,585.55|\n|1.权益法下可转损益的其他综合收益|-|-|\n|2.其他债权投资公允价值变动|-|-|\n|3.金融资产重分类计入其他综合收益的金额|-|-|\n|4.其他债权投资信用减值准备|-|-|\n|5.现金流量套期储备|-|-|\n|6.外币财务报表折算差额|-145,529.81|396,585.55|\n|7.其他|-|-|\n|归属于少数股东的其他综合收益的税后净额|-|-|\n|七、综合收益总额|-28,582,856.08|-18,188,658.29|\n|归属于母公司所有者的综合收益总额|-28,492,510.14|-18,171,673.29|\n|归属于少数股东的综合收益总额|-90,345.94|-16,985.00|\n|八、每股收益：|-|-|\n|（一）基本每股收益|-0.04|-0.03|\n|（二）稀释每股收益|-0.04|-0.03|本期发生同一控制下企业合并的，被合并方在合并前实现的净利润为：元，上期被合并方实现的净利润为：元。\n法定代表人：姚庆喜                     主管会计工作负责人：王厚强                      会计机构负责人：王厚\n强\n                 

# 平安银行股份有限公司合并利润表\n2024年 1-3月\n|null|null|货币单位：人民币百万元|\n|---|---|---|\n|项目|2024年 1-3月|2023年 1-3月|\n|一、营业收入|38,770|45,098|\n|利息净收入|25,157|32,115|\n|利息收入|53,369|58,692|\n|利息支出|(28,212)|(26,577)|\n|手续费及佣金净收入|7,181|8,878|\n|手续费及佣金收入|8,216|10,366|\n|手续费及佣金支出|(1,035)|(1,488)|\n|投资收益|4,702|3,303|\n|其中：以摊余成本计量的金融资产终止确认<br>产生的收益|-|(1)|\n|公允价值变动损益|1,572|54|\n|汇兑损益|7|315|\n|其他业务收入|87|253|\n|资产处置损益|3|7|\n|其他收益|61|173|\n|二、营业支出|(10,820)|(12,265)|\n|税金及附加|(390)|(458)|\n|业务及管理费|(10,430)|(11,807)|\n|三、减值损失前营业利润|27,950|32,833|\n|信用减值损失|(9,395)|(14,449)|\n|其他资产减值损失|(1)|(15)|\n|四、营业利润|18,554|18,369|\n|加：营业外收入|12|11|\n|减：营业外支出|(41)|(13)|\n|五、利润总额|18,525|18,367|\n|减：所得税费用|(3,593)|(3,765)|\n|六、净利润|14,932|14,602|\n|持续经营净利润|14,932|14,602|\n|终止经营净利润|-|-|\n|七、其他综合收益的税后净额|345|(448)|\n|(一)不能重分类进损益的其他综合收益|(6)|(38)|\n|其他权益工具投资公允价值变动|(6)|(38)|\n|(二)将重分类进损益的其他综合收益|351|(410)|\n|1.以公允价值计量且其变动计入其他综合收益的|-|-|\n|金融资产的公允价值变动|132|152|\n|2.以公允价值计量且其变动计入其他综合收益的<br>金融资产的信用损失准备|216|(567)|\n|3.外币财务报表折算差额|3|5|\n|八、综合收益总额|15,277|14,154|\n|九、每股收益|-|-|\n|（一）基本每股收益(元/股)|0.66|0.65|\n|（二）稀释每股收益(元/股)|0.66|0.65|副行长兼\n法定代表人                   行长                   首席财务官                   会计机构负责人\n谢永林                冀光恒                     项有志                          朱培卿

# ######################
# Output:
# ("entity"{tuple_delimiter}"海峡创新"{tuple_delimiter}"公司"){record_delimiter}                                         
# ("entity"{tuple_delimiter}"-24,488,643.93"{tuple_delimiter}"利润总额"){record_delimiter}                                         
# ("entity"{tuple_delimiter}"平安银行股份有限公司"{tuple_delimiter}"公司"){record_delimiter}                                         
# ("entity"{tuple_delimiter}"18,525 (百万元)"{tuple_delimiter}"利润总额"){record_delimiter}
# ("relationship"{tuple_delimiter}"海峡创新"{tuple_delimiter}"拥有利润总额"{tuple_delimiter}"-24,488,643.93"){record_delimiter}
# ("relationship"{tuple_delimiter}"平安银行股份有限公司"{tuple_delimiter}"拥有利润总额"{tuple_delimiter}"18,525"){completion_delimiter}         

# ###################### 
# """,
]

PROMPTS["GRAPH_DESCRIPTION"] = """  
You will receive multiple triples, follow this format: [[head1, relation1, tail1], [head2, relation2, tail2], ...] , representing specific relationships between entities.  
Your task is to transform these triples into fluent, natural descriptions that effectively convey their logical connections.  

The output language should match the text document, supporting both Chinese and English.  

######################
Graph Data: {content}
######################
Output:
"""  



# Addition Prompts
PROMPTS["TABLE_CONTINUE_PROMPT"] = """
Based on the current table and the original data, is the table complete and accurate?
Current Table:
{current_table}

Original Data:
{content}

Query:
{query}

Answer with 'yes' or 'no'.
"""

PROMPTS["TABLE_IF_LOOP_PROMPT"] = """
Does the current table need to be rechecked against the original data to ensure no information is missing or incorrect?
Current Table:
{current_table}

Original Data:
{content}

Query:
{query}

Answer strictly with 'yes' (needs recheck) or 'no' (table is acceptable).
"""

PROMPTS["TABLE_RECHECK_PROMPT"] = """\
Reconstruct the table by combining Original Data and Current Table while STRICTLY MAINTAINING the existing format structure.

# Format Requirements
1. Follow this EXACT structure:
   ("table"{tuple_delimiter}<Title>{tuple_delimiter}<Source>{tuple_delimiter}<Description>){record_delimiter}
   ("header"{tuple_delimiter}<COLUMN_1>{tuple_delimiter}<COLUMN_2>...){record_delimiter}
   ("row"{tuple_delimiter}<VALUE_1>{tuple_delimiter}<VALUE_2>...){record_delimiter}
   ...
   {completion_delimiter}

2. Strict Rules:
   - PRESERVE original column order and headers
   - USE EXISTING title/source/description unless new metadata found
   - DO NOT add/remove columns
   - ONLY modify rows with missing/conflicting data

# Input Context
Original Data:
{content}

Current Table Structure:
{current_table}

"""