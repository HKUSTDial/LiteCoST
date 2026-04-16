# For Qwen2, you can enable the long-context capabilities by following these steps.
# modify the config.json file by including the below snippet:
# access token: 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
# """
# {
#         "architectures": [
#             "Qwen2ForCausalLM"
#         ],
#         // ...
#         "vocab_size": 152064,

#         // adding the following snippets
#         "rope_scaling": {
#             "factor": 4.0,
#             "original_max_position_embeddings": 32768,
#             "type": "yarn"
#         }
#     }
# """
# For details, refer to https://huggingface.co/Qwen/Qwen2-72B-Instruct.



# export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=0,1
# python -m vllm.entrypoints.openai.api_server \
# --port 8000 \
# --served-model-name qwen-grpo \
# --gpu_memory_utilization 0.9 \
# --tensor-parallel-size 4 \
# --model "/data/liangzhuowen/download/verl/merged/fin/cost_grpo_verl/qwen2-7b-ins" \
# --trust-remote-code 


python -m vllm.entrypoints.openai.api_server \
--port 8088 \
--served-model-name llama3.2-3b-ins-grpo \
--gpu_memory_utilization 0.9 \
--tensor-parallel-size 2 \
--model "/data/liangzhuowen/projects/LiteCoST/verl/merged/llama3.2-3b-ins" \
--trust-remote-code 
