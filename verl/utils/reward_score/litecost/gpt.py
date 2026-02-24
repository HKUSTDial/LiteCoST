import requests
import time
import base64
from dotenv import dotenv_values
import os
from openai import OpenAI

current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, ".env")
config = dotenv_values(env_path)


def encode_image(image_path: str) -> str:
    """
    Encodes the image at the given path to a base64-encoded string.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The base64-encoded image string.
    """
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    return base64_image

def get_answer(text, image=None, system_prompt=None, model='gpt-4o'):

    output = ''

    api_url = config.get("GPT_URL")
    api_key = config.get("GPT_KEY")

    # print(api_url, api_key)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    if image is not None:
        base64_image = encode_image(image)
        if system_prompt is not None:
            data = {
                'model': model,
                'messages': [
                    {"role": "system", "content": system_prompt},
                    {'role': 'user', 'content': [{'type': 'text','text': text},
                    {'type': 'image_url','image_url': {'url': f"data:image/jpeg;base64,{base64_image}"}}]
                    }
                ],
                'max_tokens': 300
            }
        else:
            data = {
                'model': model,
                'messages': [
                    {'role': 'user', 'content': [{'type': 'text','text': text},
                    {'type': 'image_url','image_url': {'url': f"data:image/jpeg;base64,{base64_image}"}}]
                    }
                ],
                'max_tokens': 300
            }
    else:
        if system_prompt is not None:
            data = {
                'model': model,
                # 'temperature': 0.0,
                'messages': [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                             ],
                'max_tokens': 1024
            }
        else:
            data = {
                'model': model,
                # 'temperature': 0.0,
                'messages': [{"role": "user", "content": text}
                             ],
                'max_tokens': 1024,
            }

    for i in range(3):
        try:
            response = requests.post(api_url, headers=headers, json=data)
            response = response.json()
            output = response['choices'][0]['message']['content']
            return output
        except:
            time.sleep(5)

    return output

if __name__ == '__main__':
    answer = get_answer("美国的首都是哪里", model="gpt-4o-mini")
    print(answer)