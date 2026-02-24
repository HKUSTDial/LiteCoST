import requests
import time
from openai import OpenAI
from dotenv import dotenv_values
import os, re

current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, ".env")
config = dotenv_values(env_path)

    
def get_answer(content, system_prompt=None, model='ours-ft'):
    """
    Get the answer from the Ours model.

    Args:
        content (str): The content to be answered.
        system_prompt (str): The system prompt.
        model (str): The model to be used.

    Returns:
        str: The answer from the Ours model.
    """
    output = ''
    api_key = config.get("OURS_KEY")
    base_url = config.get("OURS_URL")
    # print(api_key, base_url)
    client = OpenAI(api_key=api_key, base_url=base_url)

    # print(content)
    # print(system_prompt)
    if system_prompt is not None:
        response = client.chat.completions.create(
            model=model,
            temperature=0.6,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            stream=False
        )

    else:
        response = client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=[
                {"role": "user", "content": content},
            ],
            stream=False
        )

    for i in range(3):
        try:
            output = response.choices[0].message.content

            return output
        except:
            time.sleep(5)

if __name__ == '__main__':
    answer = get_answer("What is the capital of the United States?", model="ours-ft")
    print(answer)