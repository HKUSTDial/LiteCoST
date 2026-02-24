import torch

def my_reward_fn(data_source, solution_str, ground_truth, extra_info=None):
  return len(solution_str)/100


def compute_custom_reward(solution, ground_truth):
    """
    实现您的具体奖励计算逻辑
    """
    # 示例：基于答案正确性的奖励
    if solution == ground_truth:
        return 1.0
    else:
        return 0.0