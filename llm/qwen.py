import requests
import time
import base64
from dotenv import dotenv_values
import os
from openai import OpenAI

current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, ".env")
config = dotenv_values(env_path)



# qwen2.5-max
def get_answer(content, system_prompt=None, model='qwen-plus'):
    """
    Get the answer from the Qwen model.

    Args:
        content (str): The content to be answered.
        system_prompt (str): The system prompt.
        model (str): The model to be used.

    Returns:
        str: The answer from the Qwen model.
    """

    output = ''

    api_url = config.get("DASHSCOPE_URL")
    api_key = config.get("DASHSCOPE_API")
    # print(api_url, api_key)
    
    client = OpenAI(api_key=api_key, base_url=api_url)
    if system_prompt is not None:
        response = client.chat.completions.create(
            model=model,
            # temperature=0.6,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            stream=False
        )
    else:
        response = client.chat.completions.create(
            model=model,
            # temperature=0.6,
            messages=[
                {"role": "user", "content": content},
            ],
            # stream=False
        ) 

    for i in range(3):
        try:
            output = response.choices[0].message.content
            return output
        except:
            time.sleep(5)

    return output

    
    
if __name__ == '__main__':
    answer = get_answer("What is the capital of the United States?", model='qwen-plus')
    print(answer)