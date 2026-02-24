"""
This file defines a series of prompt templates for structured data extraction and processing. It mainly includes:
1. Default Delimiter Definitions:
   - TUPLE_DELIMITER ("<|>"): Used to separate elements within tuples
   - RECORD_DELIMITER ("##"): Used to separate different records
   - COMPLETION_DELIMITER ("<|COMPLETE|>"): Used to mark completion status
2. Schema Construction Prompt Template (SCHEMA_CONSTRUCTION):
3. Schema Construction Examples (SCHEMA_CONSTRUCTION_EXAMPLES):
4. Graph Construction Prompt Template (GRAPH_CONSTRUCTION):
5. Graph Construction Examples (GRAPH_CONSTRUCTION_EXAMPLES):
6. Table Construction Prompt Template (TABLE_CONSTRUCTION):
7. Table Construction Examples (TABLE_CONSTRUCTION_EXAMPLES):
8. Table_Recheck Prompt Template (TABLE_RECHECK):
This file serves as the core configuration for prompt engineering, providing standardized and structured prompt templates for downstream tasks.
"""

# schema+cot
PROMPTS = {}

PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"



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