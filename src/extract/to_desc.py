import sys
import os

# # Add the project root to Python path
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
# sys.path.insert(0, project_root)

import llm as llm
from PIL import Image, ImageDraw, ImageFont
import os
from typing import Union


# def to_desc(data: str | str, question: str) -> str:
def to_desc(data: str, question: str) -> str:
    """
    Refine the input data into a more structured or concise form tailored to answer a specific question.
    The input data can be either text or an image path.

    Args:
        data (Union[str, str]): Either a text string containing raw data, or a path to an image file.
        question (str): The specific question that the refined data should address.

    Returns:
        str: A refined and structured description of the data, tailored for the question.

    Example:
        # For text input
        data = "This is a raw dataset about weather patterns from 2010 to 2020..."
        question = "What was the average temperature in 2015?"
        refined_data = to_desc(data, question)

        # For image input
        data = "path/to/image.jpg"
        question = "What objects are in this image?"
        image_description = to_desc(data, question)
    """
    
    # Check if the input is an image path
    if data.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
        # Image processing prompt
        prompt = (
            "Please provide a detailed description of the image content. "
            "Focus on the key visual elements, objects, their relationships, "
            "and any relevant details that could help answer the following question.\n"
            f"Question: {question}"
        )
        refined_data = llm.get_answer(question=prompt, image=data)
    else:
        # Text processing prompt (original logic)
        prompt = (
            "You are a data refinement assistant. Your task is to organize and summarize the provided "
            "data into a form that is concise and directly useful for answering a given question."
            f"Question: {question}\n"
            f"Data: {data}\n\n"
            "Please summarize or reorganize the data to make it suitable for answering the question."
        )
        refined_data = llm.get_answer(question=prompt)

    return refined_data


from PIL import Image, ImageDraw, ImageFont
import os


def visualize(text, path="./text.txt"):
    try:
        with open(path, 'w', encoding='utf-8') as file:
            file.write(text)
    except Exception as e:
        print(f"An error occurred: {e}")



if __name__ == "__main__":
    # Test image description
    # Example usage
    text = "这是一个示例文本，它将根据文本的长度动态生成图片的大小。这样可以保证文本的显示效果，同时避免浪费空间。"
    visualize(text)

