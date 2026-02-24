from llm import gpt, deepseek, qwen, ours, vllm_model
from llm import asy_api
from llm import global_config as config
import asyncio

def get_answer(question, image=None, system_prompt=None, model=None):

    if model is None:
        model = config.get_model()
    print(model)
    if model.startswith('gpt'):
        output = gpt.get_answer(question, image=image, system_prompt=system_prompt, model=model)
    elif model.startswith('deepseek'):
        output = deepseek.get_answer(question, system_prompt=system_prompt, model=model)
    elif model.startswith('qwen'):
        output = qwen.get_answer(question, system_prompt=system_prompt, model=model)
    elif model.startswith('ours'):
        output = ours.get_answer(question, system_prompt=system_prompt, model=model)
    else:
        output = vllm_model.get_answer(question)

    return output

async def async_get_answer(question, model=None, system_prompt=None):
    if model is None:
        model = config.get_model()
    print(model)

    return await asy_api.async_get_answer(question, model, system_prompt)