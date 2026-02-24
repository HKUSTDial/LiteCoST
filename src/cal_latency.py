import json
import os


import tiktoken
import random
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

random.seed(10086)

def token_length(text):
    return len(encoding.encode(text, disallowed_special=()))

def calc_max_latency(json_path=None):
    """
    Calculate the maximum value of the "latency" field for each element in the specified JSON file.
    :param json_path: Path to the JSON file. Defaults to
                      the file '0a2a5a6a-94fe-43e1-9c09-e5e0bc41d56b.json' in the current directory.
    :return: The maximum latency value, or None if there is no "latency" field.
    """
    if json_path is None:
        json_path = os.path.join(os.path.dirname(__file__), '0a2a5a6a-94fe-43e1-9c09-e5e0bc41d56b.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    latencies = [item.get('latency') for item in data if isinstance(item, dict) and 'latency' in item]
    if not latencies:
        return None
    return max(latencies)


# Average based on the maximum value per item
def calc_folder_max_latency(folder_paths, jsonl_path, output_path):
    """
    Aggregate the maximum latency across all JSON files in multiple folders,
    and keep, for each id, only the largest latency value. Also compute and store
    id, cot_length and the average time per token for that record.
    :param folder_paths: List of folder paths to scan
    :param jsonl_path: Path to a JSONL file used to align and read additional 'length' by id
    :param output_path: Output JSON file path
    """
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Use a dictionary to store the maximum latency for each id
    id_max_latency = {}
    total_latency = 0
    total_tokens = 0
    total_count = 0
    
    for folder_path in folder_paths:
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} does not exist, skipping.")
            continue
        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
                        continue

                for item in data:
                    if isinstance(item, dict) and 'latency' in item and 'id' in item and 'cot_length' in item:
                        item_id = item['id']
                        if item['latency'] is None:
                            continue
                        if item_id not in id_max_latency or item['latency'] > id_max_latency[item_id]['max_latency']:
                            # Compute average time per token (seconds/token)
                            avg_time_per_token = item['latency'] / item['cot_length'] if item['cot_length'] > 0 else 0
                            id_max_latency[item_id] = {
                                'id': item_id,
                                'max_latency': item['latency'],
                                'cot_length': item['cot_length'],
                                'avg_time_per_token': avg_time_per_token
                            }
                            total_latency += item['latency']
                            total_tokens += item['cot_length']
                            total_count += 1
                        
    results = list(id_max_latency.values())
    avg_tokens = total_tokens / total_count if total_count > 0 else 0
    avg_latency = total_latency / total_count if total_count > 0 else 0
    avg_time_per_token_total = total_latency / total_tokens if total_tokens > 0 else 0

    # ------- Read jsonl and align by id -------
    id_to_length = {}
    if os.path.exists(jsonl_path):
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    if 'id' in record and 'length' in record:
                        id_to_length[record['id']] = record['length']
                except Exception as e:
                    print(f"Failed to parse jsonl: {e}")
                    continue

    total_length = 0
    matched_count = 0
    for item in results:
        item_id = item['id']
        if item_id in id_to_length:
            length_val = id_to_length[item_id]
            item['length'] = length_val  # store matched length
            total_length += length_val
            matched_count += 1

    avg_length = total_length / matched_count if matched_count > 0 else 0
    avg_cost = calc_cost(f"{avg_length:.2f}", 'gpt-4o-mini')
    print(f"Average latency: {avg_latency:.2f} s")
    print(f"Overall avg time per token: {avg_time_per_token_total:.4f} s/token")
    print(f"Average length: {avg_length:.2f}")
    print(f"Average cost: {avg_cost}")


    # with open(output_path, 'w', encoding='utf-8') as f:
        # json.dump(results, f, ensure_ascii=False, indent=4)

def calc_folder_avg_latency(folder_path, output_path):
    """
    Compute the average latency for each id within all JSON files in a folder,
    and output id, average latency, average cot_length and average time per token.
    Each id keeps an averaged latency value.
    :param folder_path: Folder path containing JSON files
    :param output_path: Output JSON file path
    """
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Use a dictionary to store the latency sum and count for each id
    id_latency_stats = {}
    total_latency = 0
    total_tokens = 0
    total_count = 0
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for item in data:
                if isinstance(item, dict) and 'latency' in item and 'id' in item and 'cot_length' in item:
                    item_id = item['id']
                    if item['latency'] is None:
                        continue
                        
                    if item_id not in id_latency_stats:
                        id_latency_stats[item_id] = {
                            'id': item_id,
                            'total_latency': 0,
                            'total_cot_length': 0,
                            'count': 0
                        }
                    
                    id_latency_stats[item_id]['total_latency'] += item['latency']
                    id_latency_stats[item_id]['total_cot_length'] += item['cot_length']
                    id_latency_stats[item_id]['count'] += 1
                    
                    total_latency += item['latency']
                    total_tokens += item['cot_length']
                    total_count += 1
    
    # Compute the averages for each id
    results = []
    for stats in id_latency_stats.values():
        avg_latency = stats['total_latency'] / stats['count']
        avg_time_per_token = stats['total_latency'] / stats['total_cot_length'] if stats['total_cot_length'] > 0 else 0
        results.append({
            'id': stats['id'],
            'avg_latency': avg_latency,  # keep field name, but this is an average value
            'cot_length': stats['total_cot_length'] // stats['count'],  # average token length (integer)
            'avg_time_per_token': avg_time_per_token
        })
                        
    avg_latency = total_latency / total_count if total_count > 0 else 0
    avg_time_per_token_total = total_latency / total_tokens if total_tokens > 0 else 0
    print(f"Average latency: {avg_latency:.2f} s")
    print(f"Overall avg time per token: {avg_time_per_token_total:.4f} s/token")
    
    # with open(output_path, 'w', encoding='utf-8') as f:
        # json.dump(results, f, ensure_ascii=False, indent=4)




def read_id_and_cot_length(json_path, output_path):
    """
    Read the fields 'id' and 'cot_length' from a JSON file and save the results to another JSON file.
    If 'cot_length' is missing, compute the length of the 'cot' field via token_length.
    :param json_path: Input JSON file path
    :param output_path: Output JSON file path
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    results = []
    for item in data:
        if isinstance(item, dict) and 'id' in item:
            if 'cot_length' in item:
                cot_len = item['cot_length']
            elif 'cot' in item:
                cot_len = token_length(item['cot'])
            else:
                continue
            results.append({
                'id': item['id'],
                'cot_length': cot_len
            })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
def merge_json_files(folder_path, output_path):
    """
    Merge all JSON files in a folder into a single JSON file.
    :param folder_path: Path to the folder containing JSON files
    :param output_path: Output file path for the merged result
    """
    merged_data = []
    
    # Traverse all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # If loaded data is a list, extend; otherwise append
                    if isinstance(data, list):
                        merged_data.extend(data)
                    else:
                        merged_data.append(data)
            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")
                continue
    
    # Write the merged data to the output file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)
    
    print(f"Merging complete, processed {len(merged_data)} records")

def calc_avg_cot_length(json_path):
    """
    Calculate the average value of 'cot_length' across all items in a JSON file.
    :param json_path: Path to the JSON file
    :return: The average cot_length; None if not available
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    cot_lengths = []
    for item in data:
        if isinstance(item, dict):
            if 'cot_length' in item:
                cot_lengths.append(item['cot_length'])
            elif 'cot' in item:
                cot_lengths.append(token_length(item['cot']))
    
    if not cot_lengths:
        return None
        
    return sum(cot_lengths) / len(cot_lengths)

def calc_cost(tokens: float, model: str = "gpt-4o") -> float:
    """
    Calculate the price (USD) based on the number of tokens and the model.
    
    :param tokens: Number of tokens
    :param model: Model name, either "gpt-4o" or "gpt-4o-mini"
    :return: Corresponding cost in USD
    """
    # Price per million tokens
    prices = {
        "gpt-4o": 5.0,        # $5 / 1M
        "gpt-4o-mini": 0.6    # $0.6 / 1M
    }
    
    if model not in prices:
        raise ValueError(f"Unknown model: {model}, must be one of {list(prices.keys())}")
    
    cost = float(tokens) * (prices[model] / 1_000_000)
    return cost


if __name__ == '__main__':
    model = 'gpt-4o'
    input_folder = f"./results/Loongfin/{model}/data_structure_results"
    input_folder = [input_folder]
    jsonl_path = "./dataset/Loong/data/loong_process_structrag.jsonl"
    output_path = "./latency/Loong/max/deepseek-r1.json"

    # Ensure the output directory exists
    calc_folder_max_latency(input_folder, jsonl_path, output_path)
    # calc_folder_avg_latency(input_folder, output_path)
    

