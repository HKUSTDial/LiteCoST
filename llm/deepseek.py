import requests
import time
from openai import OpenAI
from dotenv import dotenv_values
import os, re


current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, ".env")
config = dotenv_values(env_path)



def get_answer(content, system_prompt=None, model='deepseek-r1'):
    """
    Get the answer from the DeepSeek model.

    Args:
        content (str): The content to be answered.
        system_prompt (str): The system prompt.
        model (str): The model to be used.

    Returns:
        str: The answer from the DeepSeek model.
    """

    reasoning_output = ""
    output = ''
    api_key = config.get("DEEPSEEK_KEY")
    base_url = config.get("DEEPSEEK_URL")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    # print(base_url, api_key)
    client = OpenAI(api_key=api_key, base_url=base_url)
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
            # temperature=0.6,
            messages=[
                {"role": "user", "content": content},
            ],
            stream=False
        ) 

    for i in range(3):
        try:
            #官网
            reasoning_output = response.choices[0].message.reasoning_content
            output = response.choices[0].message.content

            if '<answer>' in output:
                final_response = f"""
                <think>
                {reasoning_output}
                </think>
                {output}
                """
            else:
                final_response = f"""
                <think>
                {reasoning_output}
                </think>
                <answer>
                {output}
                </answer>
                """
            return final_response
        except:
            time.sleep(5)

    return output

if __name__ == '__main__':
    ans = get_answer("What is the value of pi?", model="deepseek-r1")
    print(ans)