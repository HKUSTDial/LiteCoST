import requests
import time
from openai import OpenAI
from dotenv import dotenv_values
import os, re

current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, ".env")
config = dotenv_values(env_path)

    
def get_answer(content, system_prompt=None, model='llama-3.1-8b-instruct'):
    """
    Get the answer from the Llama model.

    Args:
        content (str): The content to be answered.
        system_prompt (str): The system prompt.
        model (str): The model to be used.

    Returns:
        str: The answer from the Llama model.
    """
    output = ''
    api_key = config.get("LLAMA_KEY")
    base_url = config.get("LLAMA_URL")
    # print(api_key, base_url)
    client = OpenAI(api_key=api_key, base_url=base_url)

    if system_prompt is not None:
        response = client.chat.completions.create(
            model=model,
            # temperature=0.6,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            stream=True
        )

    else:
        response = client.chat.completions.create(
            model=model,
            # temperature=0.6,
            messages=[
                {"role": "user", "content": content},
            ],
            stream=True
        )

    for i in range(3):
        try:
            # output = response.choices[0].message.content

            # return output

            collected_chunks = []  # 收集流的片段

            for chunk in response:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    # print(chunk.choices[0].delta.content)
                    collected_chunks.append(chunk.choices[0].delta.content)

            output = ''.join(collected_chunks).strip()
            return output
        except:
            time.sleep(5)

if __name__ == '__main__':
    answer = get_answer("What is the capital of the United States?", model="llama-3.1-8b-instruct")
    print(answer)